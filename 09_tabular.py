from kaggle import api
from fastbook import *
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
from fastai.tabular.all import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from dtreeviz.trees import *
from IPython.display import Image, display_svg, SVG

pd.options.display.max_rows = 20
pd.options.display.max_columns = 8

creds = '{"username":"hoangminhvu","key":"834deb15e00f2a3b571f4f64180eb241"}'
cred_path = Path('~/.kaggle/kaggle.json').expanduser()
if not cred_path.exists():
    cred_path.parent.mkdir(exist_ok=True)
    cred_path.write_text(creds)
    cred_path.chmod(0o600)

comp = 'bluebook-for-bulldozers'
path = URLs.path(comp)
Path.BASE_PATH = path

api.competition_download_cli(comp, path=path)
shutil.unpack_archive(str(path/f'{comp}.zip'), str(path))


# if not path.exists():
#     path.mkdir(parents=true)
#     api.competition_download_cli(comp, path=path)
#     shutil.unpack_archive(str(path/f'{comp}.zip'), str(path))

# df = pd.read_csv(path/'TrainAndValid.csv', low_memory=False)
# df = add_datepart(df, 'saledate')
# df_test = pd.read_csv(path/'Test.csv', low_memory=False)
# df_test = add_datepart(df_test, 'saledate')

# procs = [Categorify, FillMissing]
# cond = (df.saleYear < 2011) | (df.saleMonth < 10)
# train_idx = np.where(cond)[0]
# valid_idx = np.where(~cond)[0]

# dep_var = 'SalePrice'
# splits = (list(train_idx), list(valid_idx))
# cont, cat = cont_cat_split(df, 1, dep_var=dep_var)
# to = TabularPandas(df, procs, cat, cont, y_names=dep_var, splits=splits)
