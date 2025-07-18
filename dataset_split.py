import os
import pandas as pd
from sklearn.model_selection import train_test_split

class EyeDiseaseDataset:
    def _init_(self, data_dir, img_exts=(".jpg", ".jpeg", ".png"), test_size=0.1, val_size=0.1, random_state=42):
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        self.data_dir = data_dir
        self.img_exts = img_exts
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.df = self._build_df()

    def _build_df(self):
        rows = []
        for cls in os.listdir(self.data_dir):
            cls_path = os.path.join(self.data_dir, cls)
            if not os.path.isdir(cls_path):
                continue
            for fname in os.listdir(cls_path):
                if fname.lower().endswith(self.img_exts):
                    rows.append({
                        "filepath": os.path.join(cls_path, fname),
                        "label": cls
                    })
        if not rows:
            raise RuntimeError(f"No images found under {self.data_dir}. Check folder structure.")
        return pd.DataFrame(rows)

    def split_(self):
        df = self.df.copy()
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            stratify=df["label"],
            random_state=self.random_state,
            shuffle=True,
        )
        relative_val_size = self.val_size / (1.0 - self.test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_size,
            stratify=train_val_df["label"],
            random_state=self.random_state,
            shuffle=True,
        )
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def split(self):
        return self.split_()