import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import LabelEncoder

class ProjectDataLoader:
    """
    Loads project data from CSV:
      - metadata.csv (sample_id, species, batch, treatment, ultrasound)
      - nir_data.csv (first col sample_id, then wavelengths as headers)
      - chemical_data.csv (first col sample_id, then targets)
      - particle_size.csv (first col sample_id, then PSD bins like 10µm..5000µm)
      - texture.csv (first col sample_id, then force series over index)
    Resolves duplicates by sampling ONE row per sample_id (reproducible).
    Builds labels (species, batch, treatment, ultrasound, phase) + composites.
    """

    # -------------------- init --------------------
    def __init__(self, data_dir="./data", seed=42):
        self.data_dir = Path(data_dir)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.df_meta = None
        self.df_nir = None
        self.df_chem = None
        self.df_psd = None
        self.df_tex = None

        self.df_psd_summary = None
        self.df_tex_summary = None

        self.df_joint = None
        self.X_nir = None
        self.wavelengths = None
        self.Y_chem = None

        self._cat_views = {}

    # -------------------- utils --------------------
    @staticmethod
    def _require_cols(df, required, name):
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{name}: missing required columns: {missing}")

    @staticmethod
    def _to_int_if_possible(s):
        vals = pd.to_numeric(s, errors="coerce")
        return vals.round().astype("Int64")

    @staticmethod
    def _clean_string(s):
        return (
            s.astype(str)
             .str.strip()
             .replace({"nan": pd.NA, "None": pd.NA}, regex=False)
        )

    def _sample_one_per_id(self, df, id_col="sample_id"):
        if df.empty:
            return df
        if df[id_col].is_unique:
            return df.reset_index(drop=True)
        return (
            df.groupby(id_col, as_index=False, group_keys=False)
              .apply(lambda g: g.sample(n=1, random_state=self.seed))
              .reset_index(drop=True)
        )

    # -------------------- readers --------------------
    def load_metadata(self, filename="metadata.csv"):
        df = pd.read_csv(self.data_dir / filename)
        self._require_cols(df, ["sample_id", "species", "batch", "treatment", "ultrasound"], "metadata")
        df["sample_id"] = self._clean_string(df["sample_id"])
        df["species"] = self._clean_string(df["species"])
        enc_sp = LabelEncoder()
        df['species_class'] = enc_sp.fit_transform(df['species'])
        df["treatment"] = self._clean_string(df["treatment"])
        df["batch"] = self._to_int_if_possible(df["batch"])
        df["ultrasound"] = self._to_int_if_possible(df["ultrasound"])
        self.df_meta = self._sample_one_per_id(df, "sample_id")
        

    def load_nir(self, filename="nir_data.csv"):
        df = pd.read_csv(self.data_dir / filename)
        self._require_cols(df, ["sample_id"], "nir_data")
        df["sample_id"] = self._clean_string(df["sample_id"])

        wave_cols = [c for c in df.columns if c != "sample_id"]
        wmap, bad = {}, []
        for c in wave_cols:
            s = str(c).lower().replace("nm", "").strip()
            try:
                wmap[c] = float(s)
            except:
                bad.append(c)
        if bad:
            print(f"[load_nir] Non-wavelength columns ignored: {bad}")
        df = df.drop(columns=bad).rename(columns=wmap)
        waves_sorted = sorted([c for c in df.columns if c != "sample_id"])
        df = df[["sample_id"] + waves_sorted]
        self.df_nir = df.reset_index(drop=True)
        # take wavelengths from 700 till 2050 nm
        wavelengths = np.array([c for c in self.df_nir.columns if c != "sample_id"])
        wavelengths_index = np.array([i for i,c in enumerate(wavelengths) if c>=700 and c<=2050])
        self.wavelengths_nir = wavelengths[wavelengths_index]
        self.df_nir = self.df_nir[["sample_id"] + list(self.wavelengths_nir)]
        # self.df_nir = self.df_nir.loc[:, (self.df_nir.columns >= 700) & (self.df_nir.columns <= 2150)]

    def load_chemical(self, filename="chemical_data.csv"):
        df = pd.read_csv(self.data_dir / filename)
        self._require_cols(df, ["sample_id"], "chemical_data")
        df["sample_id"] = self._clean_string(df["sample_id"])
        self.df_chem = self._sample_one_per_id(df, "sample_id")

    def _parse_um_header(self, colname: str):
        s = str(colname).lower()
        s = s.replace("μ", "u").replace("µ", "u")
        s = s.replace("um", "").replace("micron", "").replace("microns", "").strip()
        s = s.replace("mm", "000")  # crude: 1 mm -> 1000 µm
        try:
            return float(s)
        except:
            return np.nan

    def load_particle_size(self, filename="particle_size.csv"):
        df = pd.read_csv(self.data_dir / filename)
        self._require_cols(df, ["sample_id"], "particle_size")
        df["sample_id"] = self._clean_string(df["sample_id"])
        psd_cols = [c for c in df.columns if c != "sample_id"]
        um_map = {c: self._parse_um_header(c) for c in psd_cols}
        good = [c for c in psd_cols if pd.notna(um_map[c])]
        if len(good) == 0:
            raise ValueError("particle_size: could not parse any PSD bin headers to micrometers.")
        df = df[["sample_id"] + good].rename(columns=um_map)
        sorted_bins = ["sample_id"] + sorted([c for c in df.columns if c != "sample_id"])
        df = df[sorted_bins]
        self.df_psd = self._sample_one_per_id(df, "sample_id")

    def load_texture(self, filename="texture.csv"):
        df = pd.read_csv(self.data_dir / filename)
        self._require_cols(df, ["sample_id"], "texture")
        df["sample_id"] = self._clean_string(df["sample_id"])
        self.df_tex = self._sample_one_per_id(df, "sample_id")

    # -------------------- PSD / Texture summaries --------------------
    def _build_psd_summary(self):
        if self.df_psd is None or self.df_psd.empty:
            self.df_psd_summary = pd.DataFrame(columns=["sample_id", "D10_um", "D50_um", "D90_um", "span", "skew_proxy"])
            return
        bins = [c for c in self.df_psd.columns if c != "sample_id"]
        Xbins = np.array(bins, float)

        def summarize(row):
            y = np.array(row[bins], float)
            y = np.nan_to_num(y, nan=0.0)
            # convert to cumulative (0..1)
            if y.max() <= 1.0 + 1e-9:
                dens = y / max(y.sum(), 1e-12)
                cum = np.cumsum(dens)
            else:
                cum = np.cumsum(y / 100.0)
                cum = cum / max(cum.max(), 1.0)
            idx = np.argsort(Xbins)
            x = Xbins[idx]; c = cum[idx]
            q = lambda p: float(np.interp(p, c, x, left=x.min(), right=x.max()))
            D10, D50, D90 = q(0.10), q(0.50), q(0.90)
            span = (D90 - D10) / (D50 if D50 != 0 else 1.0)
            left, right = (D50 - D10), (D90 - D50)
            skewp = (right - left) / (right + left + 1e-12)
            return pd.Series(dict(D10_um=D10, D50_um=D50, D90_um=D90, span=span, skew_proxy=skewp))

        feats = self.df_psd.apply(summarize, axis=1)
        self.df_psd_summary = pd.concat([self.df_psd[["sample_id"]].reset_index(drop=True),
                                         feats.reset_index(drop=True)], axis=1)

    def _build_texture_summary(self):
        if self.df_tex is None or self.df_tex.empty:
            self.df_tex_summary = pd.DataFrame(columns=["sample_id", "Fmax", "Work", "Slope0", "Index_to_Fmax"])
            return
        force_cols = [c for c in self.df_tex.columns if c != "sample_id"]

        def summarize(row):
            f = np.array(row[force_cols], float)
            f = np.nan_to_num(f, nan=0.0)
            if np.all(~np.isfinite(f)) or np.all(f == 0):
                return pd.Series(dict(Fmax=np.nan, Work=np.nan, Slope0=np.nan, Index_to_Fmax=np.nan))
            Fmax = float(np.nanmax(f))
            x = np.arange(f.size, dtype=float)
            Work = float(np.trapz(f, x=x))
            k = min(5, max(2, f.size // 20))
            try:
                slope0 = float(np.polyfit(x[:k], f[:k], 1)[0])
            except Exception:
                slope0 = np.nan
            try:
                idx_max = int(np.nanargmax(f))
            except Exception:
                idx_max = np.nan
            return pd.Series(dict(Fmax=Fmax, Work=Work, Slope0=slope0, Index_to_Fmax=idx_max))

        feats = self.df_tex.apply(summarize, axis=1)
        self.df_tex_summary = pd.concat([self.df_tex[["sample_id"]].reset_index(drop=True),
                                         feats.reset_index(drop=True)], axis=1)

    # -------------------- labels --------------------
    @staticmethod
    def _derive_phase(series_before):
        def map_phase(v):
            if pd.isna(v): return "unknown"
            try:
                return "before" if int(v) == 1 else "after"
            except Exception:
                return "unknown"
        return series_before.map(map_phase)

    @staticmethod
    def _to_categorical(df, cols):
        out = {}
        for c in cols:
            out[c] = pd.Categorical(df[c].astype("string"))
        return out

    @staticmethod
    def _encode_cats(cat_series):
        cats = list(cat_series.categories)
        codes = cat_series.codes.to_numpy()  # -1 for NaN
        mapping = {i: v for i, v in enumerate(cats)}
        return codes, mapping

    def _add_label_columns(self, df):
        # enforce presence
        for e in ["species", "batch", "treatment", "ultrasound"]:
            if e not in df.columns:
                df[e] = pd.NA

        # clean dtypes
        df["species"] = df["species"].astype("string").str.strip()
        df["species_class"] = pd.to_numeric(df["species"], errors="coerce").round().astype("Int64")
        df["treatment"] = df["treatment"].astype("string").str.strip()
        df["batch"] = pd.to_numeric(df["batch"], errors="coerce").round().astype("Int64")
        df["ultrasound"] = pd.to_numeric(df["ultrasound"], errors="coerce").round().astype("Int64")

        # derived phase (optional for downstream use)
        df["phase"] = self._derive_phase(df["ultrasound"]).astype("string")
        df["batch_str"] = df["batch"].astype("Int64").astype("string")

        # composites
        def _comb(*cols_):
            return ("|".join([str(v) if pd.notna(v) else "NA" for v in cols_])).strip()

        df["species_treatment"] = [_comb(a,b) for a,b in zip(df["species"], df["treatment"])]
        df["species_batch"] = [_comb(a,b) for a,b in zip(df["species"], df["batch_str"])]
        df["species_treatment_phase"] = [_comb(a,b,c) for a,b,c in zip(df["species"], df["treatment"], df["phase"])]

        # cache categorical views
        self._cat_views = self._to_categorical(
            df,
            ["species","treatment","phase","batch_str","species_treatment","species_batch","species_treatment_phase"]
        )
        return df

    # -------------------- orchestrate --------------------
    def load_all(self):
        """Read all CSVs, sample one per sample_id, compute summaries, and build merged joint table + labels."""
        self.load_metadata()
        self.load_nir()
        self.load_chemical()
        self.load_particle_size()
        self.load_texture()

        # summaries
        self._build_psd_summary()
        self._build_texture_summary()

        # merge ONTO NIR (anchor)
        df = self.df_nir.copy()
        waves = [c for c in df.columns if c != "sample_id"]
        self.wavelengths = np.array(sorted(waves), float)
        df = df[["sample_id"] + list(self.wavelengths)]

        def ljoin(left, right, name):
            if right is None or right.empty:
                return left
            dup = right.columns.intersection(left.columns).difference(["sample_id"])
            if len(dup):
                right = right.rename(columns={c: f"{c}_{name}" for c in dup})
            return left.merge(right, on="sample_id", how="left")

        df = ljoin(df, self.df_meta, "meta")          # brings in labels
        df = ljoin(df, self.df_chem, "chem")
        df = ljoin(df, self.df_psd_summary, "psd")
        df = ljoin(df, self.df_tex_summary, "tex")

        # labels & composites
        df = self._add_label_columns(df)

        self.df_joint = df.reset_index(drop=True)
        self.X_nir = self.df_joint[self.wavelengths].to_numpy(float)
        
        # chemical targets
        chem_cols = [c for c in (self.df_chem.columns if self.df_chem is not None else []) if c != "sample_id"]
        self.Y_chem = self.df_joint[[c for c in self.df_joint.columns if c in chem_cols]].copy()

        # report
        print(f"[ProjectDataLoader] NIR rows: {self.df_nir.shape[0]}  | X_nir: {self.X_nir.shape}")
        print(f"[ProjectDataLoader] wavelengths: {self.wavelengths[:5]} ... {self.wavelengths[-5:]}")
        print(f"[ProjectDataLoader] labels: species, batch, treatment, ultrasound, phase (derived)")

    # -------------------- getters --------------------
    def get_joint(self):
        if self.df_joint is None:
            raise RuntimeError("Call load_all() first.")
        return self.df_joint

    def get_Xy(self, chem_targets=None):
        if self.df_joint is None:
            raise RuntimeError("Call load_all() first.")
        X = self.X_nir
        if chem_targets:
            y = self.df_joint[chem_targets].copy()
        else:
            y = self.Y_chem.copy()
        return X, y

    def get_labels(self, which=("species","treatment","batch","ultrasound","phase",
                                "species_treatment","species_batch","species_treatment_phase")):
        if self.df_joint is None:
            raise RuntimeError("Call load_all() first.")
        cols = [c for c in which if c in self.df_joint.columns]
        return self.df_joint[cols].copy()

    def get_label_arrays(self, which=("species","treatment","batch","phase","species_treatment")):
        if self.df_joint is None:
            raise RuntimeError("Call load_all() first.")
        out = {}
        for name in which:
            if name not in self.df_joint.columns:
                continue
            cat = pd.Categorical(self.df_joint[name].astype("string"))
            codes, mapping = self._encode_cats(cat)
            out[name] = (codes, mapping)
        return out

    def get_stratification_label(self, columns=("species","treatment","batch")):
        if self.df_joint is None:
            raise RuntimeError("Call load_all() first.")
        cols = [c for c in columns if c in self.df_joint.columns]
        lab = self.df_joint[cols].astype("string").fillna("NA").agg("|".join, axis=1)
        return lab.to_numpy()