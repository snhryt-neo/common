#!/usr/bin/env python3
# coding: utf-8
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials

class GspreadHandler(object):
    """ APIキーを使ってスプレッドシートを読み込むためのclass
    """
    def __init__(self, cred_json_path: str, gss_key: str):
        """ Constructor

        Args:
            cred_json_path (str):
            gss_key (str):
        """
        scope = ['https://spreadsheets.google.com/feeds']
        cred = ServiceAccountCredentials.from_json_keyfile_name(
            cred_json_path, scope
        )
        client = gspread.authorize(cred)
        self.gc = client.open_by_key(gss_key)

    def store(self, target_sheet_name: str) -> pd.DataFrame:
        """ スプレッドシートをDataFrameに格納し、それを返す
        （1行目がカラム名になっていて、indexの列がない前提）

        Args:
            target_sheet_name (str):

        Returns:
            pd.DataFrame:
        """
        worksheet = self.gc.worksheet(target_sheet_name)
        df = pd.DataFrame(worksheet.get_all_values())
        df.columns = list(df.loc[0, :])
        df.drop(0, inplace=True)
        df.reset_index(inplace=True)
        df.drop('index', axis=1, inplace=True)
        return df
