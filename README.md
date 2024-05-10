# Kaggle Leash Bio
Leash Bioの分析メモ
# 環境
データ量が大きくメモリが大きくないとそもそも分析が難しい。 

Google Colab Pro+のTPU環境を利用して分析を実施。
# コンペ概要
3つのBuilding Block + coreから大量の化合物を合成 (約1億)、3種類のタンパク質との結合を予測する分類タスク。

Building Blockのうち1つにDNAバーコードが結合されている。

コンペの肝は学習データにないBuilding Block, coreがテストデータに含まれていること。

CVとPublicが相関することや、DiscussionからPlivateに未知化合物が多く含まれることが予想される。

そのため、Publicの数値を高めるだけではShakeする可能性が高いことが予想される。
# 方針
基本的にはFP, 記述子, SMILESを利用した分析の実施。

未知化合物 (non-share)と既知化合物 (share)でCVの切り方を変える。
# 分析概要
以下に分析概要について記述する
## morgan fingerprint
データ量が大きく普通に扱うとメモリアウトするため、分割してmorgan fingerprintを実施。

得られたndarrayをTruncated SVDで次元削減して利用する。

```python
# google driveをマウント
from google.colab import drive
drive.mount('/content/drive')

# ライブラリーのインポート
!pip install rdkit
!pip install mapply
!pip install pyarrow
!pip install fastparquet

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, gc
from scipy import sparse
from tqdm import tqdm

import mapply
mapply.init( n_workers=-1,progressbar=False)

import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=(SettingWithCopyWarning))

from rdkit import Chem
from rdkit.Chem import AllChem

# ECFP変換用関数の準備

def gen_ecfp(molecule, radius=2, bits=mol_bits):
    if molecule is None:
        return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=bits))

def gen_molecule(molecule_smiles):
    if molecule_smiles is None:
        return None
    return np.array(Chem.MolFromSmiles(molecule_smiles))

def generate_ecfp_chunck(df):
    df['molecule'] = df['molecule_smiles'].mapply(gen_molecule)
    df['ecfp'] = df['molecule'].mapply(gen_ecfp)
    return sparse.csr_matrix(np.stack(df['ecfp'], axis=0).astype(np.uint8))

def generate_ecfp(df):
    n_chunks = ((len(df) - 1) // chunck_size) + 1
    ecfp_array = np.zeros((0, 2048))

    for i, _ in enumerate(tqdm(np.arange(n_chunks))):
        a=i*chunck_size
        b=i*chunck_size+chunck_size
        #print(a,b)
        #print(df_tmp.shape)
        ecfp_array = sparse.vstack([ecfp_array,generate_ecfp_chunck(df[a:b].copy())])
        if i % 100 == 0:
            sparse.save_npz(os.path.join(path, f'train_ecfp_all_{i}.npz'), ecfp_array)
            del ecfp_array
            gc.collect()
            ecfp_array = np.zeros((0, 2048))
    return ecfp_array

#変換処理
## ecfpの付与
array = generate_ecfp(df)
## 保存
sparse.save_npz(os.path.join(path, "train_ecfp_all.npz"), array)
## 終了
from google.colab import runtime
runtime.unassign()
```
