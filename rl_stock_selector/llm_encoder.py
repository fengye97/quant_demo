"""
LLM / Text Encoder for Stock Selection

Encodes financial text (news, reports, industry descriptions) into fixed-size
embedding vectors that serve as additional features for the RL agent.

Two modes are supported:
  1. REAL MODE: Uses sentence-transformers (e.g., all-MiniLM-L6-v2 or FinBERT)
     to encode stock descriptions into dense vectors.
  2. MOCK MODE: Generates reasonable embeddings from structured data (industry,
     fundamentals) without requiring an LLM. Useful for development without GPU.

Real LLM Integration Guide:
  To use with a real FinBERT / financial LLM:
    1. Install: pip install sentence-transformers
    2. Replace MockLLMEncoder with FinBERTEncoder below
    3. For Chinese financial text, consider:
       - BAAI/bge-large-zh-v1.5 (Chinese BGE)
       - jinaai/jina-embeddings-v3 (multilingual)
       - A custom fine-tuned financial BERT
    4. For production: batch-encode all stocks once per period, cache embeddings

The mock encoder uses industry one-hot + fundamental features projected to
the target dimension via PCA, providing meaningful (though not text-based)
embeddings that serve as a drop-in replacement during development.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


class BaseEncoder:
    """Abstract base class for LLM encoders."""

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode a list of text strings into embeddings.

        Parameters
        ----------
        texts : list of str
            Text descriptions to encode.

        Returns
        -------
        np.ndarray of shape (len(texts), embedding_dim)
        """
        raise NotImplementedError

    @property
    def embedding_dim(self) -> int:
        raise NotImplementedError


class MockLLMEncoder(BaseEncoder):
    """Mock encoder that generates embeddings from structured stock data.

    Uses industry one-hot encoding + fundamental feature projection
    via PCA to create a fixed-dimensional embedding for each stock.
    This provides meaningful dimensional structure without a real LLM.

    Attributes
    ----------
    embedding_dim : int
        Output embedding dimension.
    pca : sklearn.decomposition.PCA
        PCA model for dimensionality reduction.
    scaler : sklearn.preprocessing.StandardScaler
        Feature scaler.
    industry_map : dict
        Mapping from industry names to indices.
    """

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim_val = embedding_dim
        self.pca = None
        self.scaler = None
        self.industry_map = {}
        self._fitted = False

    @property
    def embedding_dim(self) -> int:
        return self.embedding_dim_val

    def fit(self, df: pd.DataFrame):
        """Fit the mock encoder on stock data.

        Uses industry categories and financial features to build
        a meaningful projection space.

        Parameters
        ----------
        df : pd.DataFrame
            Stock data with industry and fundamental columns.
        """
        # Build industry one-hot
        if "新版申万一级行业名称" in df.columns:
            industries = df["新版申万一级行业名称"].fillna("未知").unique()
            self.industry_map = {ind: i for i, ind in enumerate(sorted(industries))}

        # Gather fundamental features for PCA
        fund_cols = []
        for col in ["市盈率倒数", "市净率倒数", "涨跌幅_10", "涨跌幅_20",
                     "bias_5", "bias_10", "bias_20", "log_市值"]:
            if col in df.columns:
                fund_cols.append(col)

        if not fund_cols:
            self._fitted = True
            return

        # Prepare feature matrix
        features = df[fund_cols].copy()
        for col in fund_cols:
            features[col] = features[col].fillna(features[col].median())
            features[col] = features[col].fillna(0)
            # Clip extreme values
            q01 = features[col].quantile(0.01)
            q99 = features[col].quantile(0.99)
            features[col] = features[col].clip(q01, q99)

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(features.values)

        # PCA to target dimension
        n_components = min(self.embedding_dim_val, X.shape[1])
        self.pca = PCA(n_components=n_components, random_state=42)
        self.pca.fit(X)
        self._fitted = True

    def encode(self, df_or_texts: Union[pd.DataFrame, List[str]],
               **kwargs) -> np.ndarray:
        """Encode stocks into mock LLM embeddings.

        Parameters
        ----------
        df_or_texts : pd.DataFrame or list of str
            If DataFrame: uses structured columns to build embeddings.
            If list of str: uses text length / character features (fallback).

        Returns
        -------
        np.ndarray of shape (n_stocks, embedding_dim)
        """
        if isinstance(df_or_texts, pd.DataFrame):
            return self._encode_from_df(df_or_texts)
        else:
            return self._encode_from_texts(df_or_texts)

    def _encode_from_df(self, df: pd.DataFrame) -> np.ndarray:
        """Build embeddings from structured stock data."""
        n = len(df)

        if not self._fitted or self.pca is None:
            # Return random but deterministic embeddings
            rng = np.random.RandomState(hash(str(df.index.tolist())) % (2**31))
            emb = rng.randn(n, self.embedding_dim_val) * 0.1
            # Normalize to unit length
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)
            return emb / norms

        # Industry one-hot
        if self.industry_map and "新版申万一级行业名称" in df.columns:
            ind_matrix = np.zeros((n, len(self.industry_map)))
            for i, (_, row) in enumerate(df.iterrows()):
                ind = row.get("新版申万一级行业名称", "未知")
                if pd.isna(ind):
                    ind = "未知"
                if ind in self.industry_map:
                    ind_matrix[i, self.industry_map[ind]] = 1.0
        else:
            ind_matrix = np.zeros((n, 1))

        # Fundamental features
        fund_cols = ["市盈率倒数", "市净率倒数", "涨跌幅_10", "涨跌幅_20",
                      "bias_5", "bias_10", "bias_20"]
        fund_data = np.zeros((n, len(fund_cols)))
        for j, col in enumerate(fund_cols):
            if col in df.columns:
                vals = df[col].fillna(0).values
                fund_data[:, j] = vals

        # Scale
        if self.scaler is not None:
            try:
                fund_data = self.scaler.transform(fund_data)
            except Exception:
                pass

        # Combine and project
        combined = np.concatenate([fund_data, ind_matrix], axis=1)

        if self.pca is not None:
            try:
                # Pad or truncate to match PCA components
                if combined.shape[1] < self.pca.n_components_:
                    pad = np.zeros((n, self.pca.n_components_ - combined.shape[1]))
                    combined = np.concatenate([combined, pad], axis=1)
                elif combined.shape[1] > self.pca.n_features_in_:
                    combined = combined[:, :self.pca.n_features_in_]

                # Align with fitted feature count
                if combined.shape[1] == self.pca.n_features_in_:
                    emb = self.pca.transform(combined)
                else:
                    emb = combined[:, :self.pca.n_components_]
            except Exception:
                emb = np.zeros((n, self.pca.n_components_))
        else:
            emb = np.zeros((n, self.embedding_dim_val))

        # Pad to target embedding_dim
        if emb.shape[1] < self.embedding_dim_val:
            pad = np.zeros((n, self.embedding_dim_val - emb.shape[1]))
            emb = np.concatenate([emb, pad], axis=1)
        elif emb.shape[1] > self.embedding_dim_val:
            emb = emb[:, :self.embedding_dim_val]

        # Normalize
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        emb = emb / norms

        return emb.astype(np.float32)

    def _encode_from_texts(self, texts: List[str]) -> np.ndarray:
        """Fallback: generate embeddings from raw text strings."""
        n = len(texts)
        rng = np.random.RandomState(42)
        emb = np.zeros((n, self.embedding_dim_val), dtype=np.float32)

        for i, text in enumerate(texts):
            # Use hash of text for deterministic pseudo-embedding
            h = hash(text) % (2**31)
            local_rng = np.random.RandomState(h)
            emb[i] = local_rng.randn(self.embedding_dim_val).astype(np.float32) * 0.1

        # Normalize
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return emb / norms


class FinBERTEncoder(BaseEncoder):
    """Real LLM encoder using sentence-transformers.

    This is the production encoder. It requires:
      pip install sentence-transformers

    Recommended models:
      - "BAAI/bge-large-zh-v1.5"  (Chinese, 1024-dim, best quality)
      - "all-MiniLM-L6-v2"         (English, 384-dim, lightweight)
      - "ProsusAI/finbert"         (English financial BERT, 768-dim)

    For Chinese financial text, BGE is the best open-source option.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 device: str = "cpu"):
        """
        Parameters
        ----------
        model_name : str
            HuggingFace model name for sentence-transformers.
        device : str
            "cpu" or "cuda".
        """
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def embedding_dim(self) -> int:
        if self._model is None:
            self._load_model()
        return self._model.get_sentence_embedding_dimension()

    def _load_model(self):
        """Lazy-load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for FinBERTEncoder. "
                    "Install with: pip install sentence-transformers"
                )

    def encode(self, texts: List[str],
               batch_size: int = 32,
               show_progress_bar: bool = False,
               **kwargs) -> np.ndarray:
        """Encode text descriptions into embeddings.

        Parameters
        ----------
        texts : list of str
            Text descriptions of stocks (e.g., industry + fundamentals).
        batch_size : int
            Encoding batch size.
        show_progress_bar : bool
            Show tqdm progress bar.

        Returns
        -------
        np.ndarray of shape (len(texts), embedding_dim)
        """
        if self._model is None:
            self._load_model()

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
            **kwargs
        )
        return embeddings.astype(np.float32)


def build_stock_descriptions(df: pd.DataFrame, date: pd.Timestamp = None) -> List[str]:
    """Build text descriptions for each stock from structured data.

    These descriptions simulate what a financial analyst might read about a stock,
    and can be encoded by an LLM to produce semantic embeddings.

    Parameters
    ----------
    df : pd.DataFrame
        Stock data for a single date (or all dates).
    date : pd.Timestamp, optional
        If given, filter to this date.

    Returns
    -------
    list of str
        Stock descriptions.
    """
    if date is not None:
        df = df[df["交易日期"] == date]

    descriptions = []
    for _, row in df.iterrows():
        parts = []

        # Stock name
        name = row.get("股票名称", "未知")
        code = str(row.get("股票代码", ""))
        parts.append(f"股票{name}({code})")

        # Industry
        industry = row.get("新版申万一级行业名称", "")
        if pd.notna(industry) and industry:
            parts.append(f"所属{industry}行业")
        ind2 = row.get("新版申万二级行业名称", "")
        if pd.notna(ind2) and ind2:
            parts.append(f"细分{ind2}")

        # Valuation
        pe_inv = row.get("市盈率倒数", np.nan)
        pb_inv = row.get("市净率倒数", np.nan)
        if pd.notna(pe_inv) and pe_inv > 0:
            parts.append(f"盈利收益率{pe_inv:.4f}")
        if pd.notna(pb_inv) and pb_inv > 0:
            parts.append(f"净资产收益率倒数{pb_inv:.4f}")

        # Momentum
        ret_10 = row.get("涨跌幅_10", np.nan)
        ret_20 = row.get("涨跌幅_20", np.nan)
        if pd.notna(ret_10):
            direction = "上涨" if ret_10 > 0 else "下跌"
            parts.append(f"近10日{direction}{abs(ret_10):.2%}")
        if pd.notna(ret_20):
            direction = "上涨" if ret_20 > 0 else "下跌"
            parts.append(f"近20日{direction}{abs(ret_20):.2%}")

        desc = "，".join(parts) + "。"
        descriptions.append(desc)

    return descriptions


def get_encoder(mode: str = "mock",
                embedding_dim: int = 64,
                model_name: str = "all-MiniLM-L6-v2",
                device: str = "cpu",
                **kwargs) -> BaseEncoder:
    """Factory function to create an LLM encoder.

    Parameters
    ----------
    mode : str
        "mock" or "finbert" (or "real").
    embedding_dim : int
        Output dimension (mock mode only).
    model_name : str
        HuggingFace model name (finbert mode only).
    device : str
        "cpu" or "cuda" (finbert mode only).

    Returns
    -------
    BaseEncoder
    """
    if mode in ("finbert", "real"):
        return FinBERTEncoder(model_name=model_name, device=device)
    else:
        return MockLLMEncoder(embedding_dim=embedding_dim)
