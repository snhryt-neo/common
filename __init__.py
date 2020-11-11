#!/usr/bin/env python3
# coding: utf-8
import os
import time
import datetime as dt
import numpy as np
import pandas as pd
import pymongo
from typing import List, Dict, Any, Union
from contextlib import contextmanager
from dateutil.relativedelta import relativedelta

def date_range(
        start_date: dt.date,
        stop_date: dt.date,
        step: relativedelta = relativedelta(days=+1)
    ) -> dt.date:
    """ dt.date 型でfor文使うためのイテレーター

    Args:
        start_date (dt.date):
        stop_date (dt.date):
        step (relativedelta, optional): Defaults to relativedelta(days=+1).

    Yields:
        Iterator[dt.date]:
    """
    current_date = start_date
    while current_date < stop_date:
        yield current_date
        current_date += step

def get_first_date_this_month(input_date: dt.date) -> dt.date:
    """ `input_date` の月の月初（1日）を dt.date 型で返す。

    Args:
        input_date (dt.date):

    Returns:
        dt.date:
    """
    return dt.date(input_date.year, input_date.month, 1)

def get_last_date_this_month(input_date: dt.date) -> dt.date:
    """ `input_date` の月の月末の日付を dt.date 型で返す。

    Args:
        input_date (dt.date):

    Returns:
        dt.date:
    """
    return input_date + relativedelta(months=+1, day=1, days=-1)

def utc2jst(input_datetime: dt.datetime) -> dt.datetime:
    """ UTC表記の dt.datetime をJST表記に変換する

    Args:
        input_datetime (dt.datetime):

    Returns:
        dt.datetime:
    """
    return pd.to_datetime(input_datetime, utc=True).tz_convert('Asia/Tokyo')

def append_list_in_dict(
        input_dict: Dict[Any, List[Any]],
        *,
        key: Any,
        value: Any,
    ) -> Dict[Any, List[Any]]:
    """ valueがlist型である辞書に対して、そのlist型valueに要素をappendする。
    もし `key` が辞書内に存在しない場合は、新しいkey, valueの対応を追加する。

    Examples:
        sample_dict = `{'a': [1, 2, 3]}`
        - append_list_in_dict(sample_dict, key='a', value=1)
            -> `{'a': [1, 2, 3, 1]}`
        - append_list_in_dict(sample_dict, key='b', value=1)
            -> `{'a': [1, 2, 3], 'b': [1]}`

    Args:
        input_dict (Dict[Any, List[Any]]):
        key (Any):
        value (Any):

    Returns:
        Dict[Any, List[Any]]:
    """
    if key in input_dict.keys():
        input_dict[key].append(value)
    else:
        input_dict[key] = [value]
    return input_dict

def get_dict_value(
        input_dict: Dict[str, Any],
        key: str,
        value_when_empty: Any = np.nan,
        list_join: bool = False,
        separator: str = ';'
    ) -> Any:
    """ `input_dict` の `key` に対応するvalueを返す。
    もし `key` が存在しない場合には `value_when_empty` を返す。

    Args:
        input_dict (Dict[str, Any]):
        key (str):
        value_when_empty (Any, optional): Defaults to np.nan.
        list_join (bool, optional): Defaults to False.
        separator (str, optional): Defaults to ';'.

    Returns:
        Any:
    """
    if key in input_dict.keys():
        value = input_dict[key]
        if type(value) == list and list_join: return separator.join(value)
        else: return value
    else:
        return value_when_empty

def init_dict_for_df(df_columns: List[str]) \
    -> Dict[str, List[None]]:
    """ そのままpd.DataFrame型に変換できるように、key: カラム名、value: 空配列
    の辞書を作成し、それを返す。

    Args:
        df_columns (List[str]):

    Returns:
        Dict[str, List[None]]:
    """
    return {column: [] for column in df_columns}

def sort_dataframe(
        df: pd.DataFrame,
        main_sort_column_index: int,
        *args: int,
        ascending: bool=True
    ) -> pd.DataFrame:
    """ pd.DataFrame 型をソートする

    Args:
        df (pd.DataFrame):
        main_sort_column_index (int):
        ascending (bool, optional): Defaults to True.

    Returns:
        pd.DataFrame:
    """
    columns = df.columns
    sort_column_names = [columns[main_sort_column_index]]
    if args:
        for index in args:
            if index >= len(columns): continue
            sort_column_names.append(columns[index])
    sorted_df = df.sort_values(
        sort_column_names, ascending=ascending).reset_index(drop=True)
    return sorted_df

def get_str2date_converted_df(
        df: pd.DataFrame,
        columns: Union[str, List[str]]
    ) -> pd.DataFrame:
    """ `df` の指定列をstr型からdt.date型に変換したものを返す。
    `columns` がlist型のときは複数列を変換する。

    Args:
        df (pd.DataFrame):
        columns (Union[str, List[str]]):

    Raises:
        KeyError: `df` 内に該当列がないとき

    Returns:
        pd.DataFrame:
    """
    df_copy = df.copy()
    if type(columns) == str: columns = [columns]
    for column in columns:
        if column in df.columns.tolist():
            df_copy[column] = df[column].map(
                lambda x: pd.to_datetime(x).date() \
                    if isinstance(x, str) else pd.NaT
            )
        else:
            raise KeyError(column)
    return df_copy

def write_csv_for_debug(
        data_dict: Dict[str, List[Any]],
        output_dirpath: str='./',
        output_filename: str='debug.csv'
    ) -> None:
    """ デバッグ用に辞書オブジェクトをcsvに書き出す

    Args:
        data_dict (Dict[str, List[Any]]):
        output_dirpath (str, optional): Defaults to './'.
        output_filename (str, optional): Defaults to 'debug.csv'.
    """
    if output_filename == 'debug.csv':
        now = dt.datetime.now()
        now_str = now.strftime('%Y%m%d-%H%M%S')
        output_filename = now_str + '_' + output_filename
    os.makedirs(output_dirpath, exist_ok=True)
    output_filepath = os.path.join(output_dirpath, output_filename)

    keys = list(data_dict.keys())
    n_keys = len(keys)
    max_length = max([len(data_dict[key]) for key in keys])
    with open(output_filepath, mode='w') as f:
        header = ','.join([f'{key}' for key in keys])
        header += '\n'
        f.write(header)
        for i in range(max_length):
            line = ''
            for j, key in enumerate(keys):
                line += f'"{data_dict[key][i]}",' \
                    if i < len(data_dict[key]) else '"",'
            line = line[: -1] + '\n'
            f.write(line)

def get_diff_df(
        original_df: pd.DataFrame,
        new_df: pd.DataFrame,
        mode: str = 'before',
        ignore_column_indices: List[int] = []
    ) -> pd.DataFrame:
    """ 同一のカラムをもつ新旧2つのpd.DataFrameを比較する。
    （一度2つを結合してから比較を行うため、新旧で行のindexが変わっていても問題ない）
    `mode` が'before'の場合は、古いほうのpd.DataFrameで更新された行を抽出し、
    その行だけから成るpd.DataFrameを返す。
    'after'の場合は、新しいほうのpd.DataFrameで更新・追加されている行を抽出し、
    その行だけから成るpd.DataFrameを返す。
    `ignore_column_indices` が指定されている場合には、該当のカラムは比較対象から除外する。

    Args:
        original_df (pd.DataFrame):
        new_df (pd.DataFrame):
        mode (str, optional): Defaults to 'before'.
        ignore_column_indices (List[int], optional): Defaults to [].

    Raises:
        KeyError: `original_df` と `new_df` のカラム数が違う場合

    Returns:
        pd.DataFrame:
    """
    if len(original_df.columns) != len(new_df.columns):
        raise KeyError(
            f'original_df has {len(original_df.columns)} columns, ' \
                + f'but new_df has {len(new_df.columns)} columns'
        )

    df_columns = list(original_df.columns)
    if len(ignore_column_indices) == 0:
        target_columns = df_columns
    else:
        target_columns = [
            df_columns[i] for i in range(len(df_columns)) \
                if not i in ignore_column_indices
        ]
    extracted_original_df = original_df[target_columns]
    extracted_new_df = new_df[target_columns]
    start_index = original_df.index.max() + 1
    end_index = len(new_df) + start_index
    extracted_new_df.index = [i for i in range(start_index, end_index)]
    df = pd.concat([extracted_original_df, extracted_new_df])
    df_gpby = df.groupby(df.columns.tolist())
    row_indices = [
        indices[0] for indices in df_gpby.groups.values() if len(indices) == 1
    ]
    if mode == 'after':
        diff_row_indices = [
            index - start_index for index in row_indices if index >= start_index
        ]
        return new_df[new_df.index.isin(diff_row_indices)]
    else:
        diff_row_indices = [
            index for index in row_indices if index < start_index
        ]
        return original_df[original_df.index.isin(diff_row_indices)]

@contextmanager
def show_processing_time():
    """ Show processing time.
    """
    start = time.time()
    try:
        yield
    finally:
        processing_time = time.time() - start
        if processing_time < 60 * 2:
            print(f'\n** Processing time: {processing_time:.5f} s')
        elif processing_time < 60 * 60:
            minute = int(processing_time / 60)
            print(f'\n** Processing time: {minute} m')
        else:
            hour = int(processing_time / (60 * 60))
            minute = int((processing_time - (hour * 60 * 60)) / 60)
            print(f'\n** Processing time: {hour} h, {minute} m')

def connect_mongo(
        client_uri: str,
        account: str,
        password: str,
        db_name: str
    ) -> pymongo.collection:
    """ MONGO DBに接続し、そのDB情報を pymongo.collection 型で返す

    Args:
        client_uri (str):
        account (str):
        password (str):
        db_name (str):

    Returns:
        pymongo.collection:
    """
    client = pymongo.MongoClient(client_uri)
    db = client[db_name]
    db.authenticate(name=account, password=password)
    return db
