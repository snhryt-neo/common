#!/usr/bin/env python3
# coding: utf-8
import os
import pandas as pd
from google.oauth2 import service_account

class GBQmodules(object):
    """ BigQuery上のテーブルデータを読み込む・テーブルにデータを書き込むためのclass
    """
    def __init__(self, project_id: str, credential_json_filepath: str) -> None:
        """

        Args:
            project_id (str):_
            credential_json_filepath (str):
        """
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_json_filepath
        self.project_id = project_id
        self.credentials \
            = service_account.Credentials.from_service_account_file(
                credential_json_filepath
            )

    def read(self, query: str) -> pd.DataFrame:
        """ BigQuery上のテーブルデータに対してSQLクエリを実行した結果を読み込む

        Args:
            query (str): （例）'SELECT * FROM `hoge.hoge`'

        Returns:
            pd.DataFrame:
        """
        df = pd.read_gbq(
            query,
            project_id=self.project_id,
            progress_bar_type='tqdm'
        )
        return df

    def replace(
            self,
            new_df: pd.DataFrame,
            dataset: str,
            table: str,
            progress_bar: bool = True
        ) -> None:
        """ BigQuery上に新規テーブルを作成。
        同名のテーブルがデータセット内に存在する場合は上書きする。

        Args:
            new_df (pd.DataFrame):
            dataset (str):
            table (str):
            progress_bar (bool, optional): Defaults to True.
        """
        new_df.to_gbq(
            f'{dataset}.{table}',
            project_id=self.project_id,
            if_exists='replace',
            progress_bar=progress_bar
        )

    def append(
            self,
            append_df: pd.DataFrame,
            dataset: str,
            table: str,
            progress_bar: bool = False
        ) -> None:
        """ BigQuery上に新規テーブルを作成。
        同名のテーブルがデータセット内に存在する場合は書き足す。

        Args:
            append_df (pd.DataFrame):
            dataset (str):
            table (str):
            progress_bar (bool, optional): Defaults to False.
        """
        append_df.to_gbq(
            f'{dataset}.{table}',
            project_id=self.project_id,
            if_exists='append',
            progress_bar=progress_bar
        )
