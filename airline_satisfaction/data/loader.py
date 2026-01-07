import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class DataLoader:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.df = None
        self.df_test = None
        self.label_encoder = LabelEncoder()

    def load_data(self, train_file='train.csv', test_file='test.csv'):
        if self.data_path is None:
            import kagglehub
            path = kagglehub.dataset_download("teejmahal20/airline-passenger-satisfaction")
            self.data_path = path

        csv_train = os.path.join(self.data_path, train_file)
        csv_test = os.path.join(self.data_path, test_file)

        self.df = pd.read_csv(csv_train)
        self.df_test = pd.read_csv(csv_test)

        self._clean_column_names()
        return self.df, self.df_test

    def _clean_column_names(self):
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_').str.replace('/', '_')
        if self.df_test is not None:
            self.df_test.columns = self.df_test.columns.str.lower().str.replace(' ', '_').str.replace('/', '_')

    def get_column_groups(self):
        cat_cols = ['gender', 'customer_type', 'type_of_travel', 'class']
        num_cols = ['age', 'flight_distance', 'departure_delay_in_minutes', 'arrival_delay_in_minutes']
        rating_cols = ['inflight_wifi_service', 'departure_arrival_time_convenient',
                       'ease_of_online_booking', 'gate_location', 'food_and_drink',
                       'online_boarding', 'seat_comfort', 'inflight_entertainment',
                       'on-board_service', 'leg_room_service', 'baggage_handling',
                       'checkin_service', 'inflight_service', 'cleanliness']
        target_col = 'satisfaction'

        return cat_cols, num_cols, rating_cols, target_col

    def reclass_variables(self):
        cat_cols, num_cols, rating_cols, _ = self.get_column_groups()

        for col in cat_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype('category')

        for col in rating_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype('category')

    def prepare_features_target(self):
        _, _, _, target_col = self.get_column_groups()

        if 'id' in self.df.columns:
            self.df = self.df.drop('id', axis=1)

        if 'unnamed:_0' in self.df.columns:
            self.df = self.df.drop('unnamed:_0', axis=1)

        X = self.df.drop(target_col, axis=1)
        y = self.df[target_col]

        y_encoded = self.label_encoder.fit_transform(y)

        return X, y_encoded

    def split_data(self, X, y, test_size=0.25, val_size=0.2, random_state=12345):
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size,
            stratify=y_train_val, random_state=random_state
        )

        return X_train, X_val, X_test, y_train, y_val, y_test, X_train_val, y_train_val

    def create_sample(self, X_train, y_train, sample_size=0.4, random_state=12345):
        X_sample, _, y_sample, _ = train_test_split(
            X_train, y_train,
            train_size=sample_size,
            stratify=y_train,
            random_state=random_state
        )
        return X_sample, y_sample

    def get_label_mapping(self):
        return dict(zip(self.label_encoder.classes_,
                       self.label_encoder.transform(self.label_encoder.classes_)))
