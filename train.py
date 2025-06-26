'''
Simple MLP with xavier initialization, dropout, and early stopping first attempt
KAGGLE House Proces score: 0.13805 (2065/4699 on leaderboard)
'''
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import zipfile
import os
import argparse


class MLPBlock(nn.Module):
    """
    A single MLP block with linear layer, activation, and optional dropout/batch norm
    """
    def __init__(self, input_dim, output_dim, activation='relu', dropout=0.0,
                 batch_norm=False, init_method='xavier_uniform'):
        super(MLPBlock, self).__init__()
        
        self.linear = nn.Linear(input_dim, output_dim)
        
        #initialize weights
        self._initialize_weights(init_method, activation)
        
        #activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()
        
        #optional batch normalization
        self.batch_norm = nn.BatchNorm1d(output_dim) if batch_norm else None
        
        #optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def _initialize_weights(self, init_method, activation):
        """Initialize weights based on the specified method"""
        if init_method == 'xavier_uniform':
            nn.init.xavier_uniform_(self.linear.weight)
        elif init_method == 'xavier_normal':
            nn.init.xavier_normal_(self.linear.weight)
        elif init_method == 'kaiming_uniform':
            nn.init.kaiming_uniform_(self.linear.weight, nonlinearity=activation)
        elif init_method == 'kaiming_normal':
            nn.init.kaiming_normal_(self.linear.weight, nonlinearity=activation)
        elif init_method == 'orthogonal':
            nn.init.orthogonal_(self.linear.weight)
        #default PyTorch initialization if none specified
        
        #initialize bias to zero (common practice)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)
    
    def forward(self, x):
        x = self.linear(x)
        
        if self.batch_norm:
            x = self.batch_norm(x)
        
        x = self.activation(x)
        
        if self.dropout:
            x = self.dropout(x)
        
        return x


class StackableMLP(nn.Module):
    """
    Easy-to-configure MLP that stacks layers using nn.Sequential
    """
    def __init__(self, input_dim, hidden_dims, output_dim, 
                 activation='relu', dropout=0.0, batch_norm=False, 
                 output_activation=None, init_method='xavier_uniform'):
        super(StackableMLP, self).__init__()
        
        layers = []
        
        #input layer
        current_dim = input_dim
        
        #hidden layers
        for hidden_dim in hidden_dims:
            layers.append(MLPBlock(
                current_dim, hidden_dim, 
                activation=activation, 
                dropout=dropout, 
                batch_norm=batch_norm,
                init_method=init_method
            ))
            current_dim = hidden_dim
        
        #output layer (no activation by default)
        output_layer = nn.Linear(current_dim, output_dim)
        self._initialize_output_layer(output_layer, init_method, activation)
        layers.append(output_layer)
        
        #optional output activation
        if output_activation:
            if output_activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif output_activation == 'softmax':
                layers.append(nn.Softmax(dim=1))
            elif output_activation == 'tanh':
                layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
    
    def _initialize_output_layer(self, layer, init_method, activation):
        """Initialize the output layer weights"""
        if init_method == 'xavier_uniform':
            nn.init.xavier_uniform_(layer.weight)
        elif init_method == 'xavier_normal':
            nn.init.xavier_normal_(layer.weight)
        elif init_method == 'kaiming_uniform':
            nn.init.kaiming_uniform_(layer.weight, nonlinearity=activation)
        elif init_method == 'kaiming_normal':
            nn.init.kaiming_normal_(layer.weight, nonlinearity=activation)
        elif init_method == 'orthogonal':
            nn.init.orthogonal_(layer.weight)
        
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def download_data(path):
    # with zipfile.ZipFile('house-prices-advanced-regression-techniques.zip', 'r') as zip_ref:
    #     zip_ref.extractall('house_prices_data')

    # # List the extracted files
    # print("Extracted files:")
    # for file in os.listdir('house_prices_data'):
    #     print(file)

    train = pd.read_csv(f"{path}/train.csv")
    test = pd.read_csv(f"{path}/test.csv")
    return train, test


def preprocess_data(train_df, test_df=None, fit_scalers=True, validation_size=0.2, random_state=42):
    """
    Preprocess housing data with proper handling of train/test splits and validation set creation
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe (optional)
        fit_scalers: Whether to fit new scalers (True for training, False for inference)
        validation_size: Fraction of training data to use for validation (0.0 to 1.0)
        random_state: Random seed for reproducible splits
    
    Returns:
        If test_df is None and validation_size > 0: 
            (train_features, train_target, val_features, val_target, scalers_dict)
        If test_df is None and validation_size = 0:
            (processed_features, target, scalers_dict)
        If test_df provided and validation_size > 0:
            (train_features, train_target, val_features, val_target, test_features, scalers_dict)
        If test_df provided and validation_size = 0:
            (train_features, target, test_features, scalers_dict)
    """
    #Store scalers for later use
    scalers_dict = {}
    
    #Separate features and target from training data
    label = "SalePrice"
    original_features = train_df.drop(columns=['Id', label]).copy()
    original_target = train_df[label].copy() 
    
    #Split training data into train and validation sets if requested
    if validation_size > 0 and original_target is not None:
        train_features, val_features, train_target, val_target = train_test_split(
            original_features, 
            original_target, 
            test_size=validation_size, 
            random_state=random_state,
            stratify=None
        )
        print(f"Split training data: {len(train_features)} train, {len(val_features)} validation samples")
    else:
        train_features = original_features
        train_target = original_target
        val_features = None
        val_target = None
    
    #Handle test data if provided
    if test_df is not None:
        test_features = test_df.drop(columns=['Id']).copy()
        # missing_cols = set(train_features.columns) - set(test_features.columns)
        # for col in missing_cols:
        #     test_features[col] = 0
    
    #Identify feature types
    numeric_features = train_features.select_dtypes(include=[np.number]).columns
    categorical_features = train_features.select_dtypes(include=['object']).columns
    
    print(f"Found {len(numeric_features)} numeric features")
    print(f"Found {len(categorical_features)} categorical features")
    
    #fill missing numerical values with median
    if fit_scalers:
        numeric_fill_values = train_features[numeric_features].median()
        scalers_dict['numeric_fill_values'] = numeric_fill_values
    else:
        numeric_fill_values = scalers_dict['numeric_fill_values']
    
    train_features[numeric_features] = train_features[numeric_features].fillna(numeric_fill_values)
    if val_features is not None:
        val_features[numeric_features] = val_features[numeric_features].fillna(numeric_fill_values)
    if test_df is not None:
        test_features[numeric_features] = test_features[numeric_features].fillna(numeric_fill_values)
    
    # Fill missing categorical values with 'Unknown'
    # train_features[categorical_features] = train_features[categorical_features].fillna('Unknown')
    # if val_features is not None:
    #     val_features[categorical_features] = val_features[categorical_features].fillna('Unknown')
    # if test_df is not None:
    #     test_features[categorical_features] = test_features[categorical_features].fillna('Unknown')
    
    if fit_scalers:
        #FIXED: Ensure one-hot encoding creates numeric columns
        train_encoded = pd.get_dummies(train_features[categorical_features], prefix_sep='_', dtype=np.float32)
        scalers_dict['categorical_columns'] = train_encoded.columns.tolist()
        
        #replace categorical columns with encoded versions
        train_features = train_features.drop(columns=categorical_features)
        train_features = pd.concat([train_features, train_encoded], axis=1)
        
        #ensure all columns are numeric
        train_features = train_features.astype(np.float32)
        
        #handle validation set encoding
        if val_features is not None:
            val_encoded = pd.get_dummies(val_features[categorical_features], prefix_sep='_', dtype=np.float32)
            val_features = val_features.drop(columns=categorical_features)
            
            #ensure validation has same columns as train
            for col in scalers_dict['categorical_columns']:
                if col not in val_encoded.columns:
                    val_encoded[col] = 0.0  # Use float instead of int
            
            val_encoded = val_encoded[scalers_dict['categorical_columns']]
            val_features = pd.concat([val_features, val_encoded], axis=1)
            val_features = val_features.astype(np.float32)  # Ensure numeric
        
        #handle test set encoding
        if test_df is not None:
            test_encoded = pd.get_dummies(test_features[categorical_features], prefix_sep='_', dtype=np.float32)
            test_features = test_features.drop(columns=categorical_features)
            
            #ensure test has same columns as train
            for col in scalers_dict['categorical_columns']:
                if col not in test_encoded.columns:
                    test_encoded[col] = 0.0  # Use float instead of int
            
            test_encoded = test_encoded[scalers_dict['categorical_columns']]
            test_features = pd.concat([test_features, test_encoded], axis=1)
            test_features = test_features.astype(np.float32)  # Ensure numeric
    
    else:
        #use saved categorical columns for inference
        train_encoded = pd.get_dummies(train_features[categorical_features], prefix_sep='_', dtype=np.float32)
        train_features = train_features.drop(columns=categorical_features)
        
        #align with saved columns
        for col in scalers_dict['categorical_columns']:
            if col not in train_encoded.columns:
                train_encoded[col] = 0.0
        train_encoded = train_encoded[scalers_dict['categorical_columns']]
        train_features = pd.concat([train_features, train_encoded], axis=1)
        train_features = train_features.astype(np.float32)
    
    if fit_scalers:
        #fit scaler on training data only
        scaler = StandardScaler()
        train_features[numeric_features] = scaler.fit_transform(train_features[numeric_features])
        scalers_dict['standard_scaler'] = scaler
        
        #transform validation and test data
        if val_features is not None:
            val_features[numeric_features] = scaler.transform(val_features[numeric_features])
        if test_df is not None:
            test_features[numeric_features] = scaler.transform(test_features[numeric_features])
    
    else:
        #use saved scaler for inference
        scaler = scalers_dict['standard_scaler']
        train_features[numeric_features] = scaler.transform(train_features[numeric_features])
    
    #ensure consistent column order
    if fit_scalers:
        scalers_dict['final_columns'] = train_features.columns.tolist()
    else:
        train_features = train_features[scalers_dict['final_columns']]
    
    #ensure validation and test have same column order
    if val_features is not None:
        val_features = val_features[scalers_dict['final_columns']]
    if test_df is not None:
        test_features = test_features[scalers_dict['final_columns']]
    
    #FINAL CHECK: Print data types to verify everything is numeric
    print("Final data types check:")
    print(f"Train features dtype: {train_features.dtypes.unique()}")
    if val_features is not None:
        print(f"Val features dtype: {val_features.dtypes.unique()}")
    if test_df is not None:
        print(f"Test features dtype: {test_features.dtypes.unique()}")
    
    #return appropriate tuple based on what was provided
    if test_df is not None:
        if validation_size > 0:
            return train_features, train_target, val_features, val_target, test_features, scalers_dict
        else:
            return train_features, train_target, test_features, scalers_dict
    else:
        if validation_size > 0:
            return train_features, train_target, val_features, val_target, scalers_dict
        else:
            return train_features, train_target, scalers_dict
    

def print_raw_data_stats(train_df, test_df=None):
    """Print comprehensive statistics for raw data"""
    print("="*80)
    print("RAW DATA STATISTICS")
    print("="*80)
    
    # Basic info
    print(f"\n📊 DATASET SHAPE:")
    print(f"Training data: {train_df.shape}")
    if test_df is not None:
        print(f"Test data: {test_df.shape}")
    
    # Column types
    label = "SalePrice"
    features = train_df.drop(columns=['Id', label] if label in train_df.columns else ['Id'])
    
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    categorical_cols = features.select_dtypes(include=['object']).columns
    
    print(f"\n📈 FEATURE TYPES:")
    print(f"Numerical features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    print(f"Total features: {len(features.columns)}")
    
    # Missing values analysis
    print(f"\n❌ MISSING VALUES:")
    missing_train = train_df.isnull().sum()
    missing_train_pct = (missing_train / len(train_df)) * 100
    
    print("Training data missing values (top 10):")
    missing_summary = pd.DataFrame({
        'Column': missing_train.index,
        'Missing_Count': missing_train.values,
        'Missing_Percentage': missing_train_pct.values
    }).sort_values('Missing_Count', ascending=False).head(10)
    
    for _, row in missing_summary.iterrows():
        if row['Missing_Count'] > 0:
            print(f"  {row['Column']}: {row['Missing_Count']} ({row['Missing_Percentage']:.1f}%)")
    
    if test_df is not None:
        missing_test = test_df.isnull().sum().sum()
        print(f"Test data total missing values: {missing_test}")
    
    # Numerical features statistics
    print(f"\n📊 NUMERICAL FEATURES STATISTICS:")
    if len(numeric_cols) > 0:
        numeric_stats = features[numeric_cols].describe()
        print(f"Sample of numerical features (first 5):")
        print(numeric_stats.iloc[:, :5].round(2))
        
        # Check for extreme values
        print(f"\nNumerical features with extreme values:")
        for col in numeric_cols[:10]:  # Check first 10 numeric columns
            q1 = features[col].quantile(0.25)
            q3 = features[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = features[(features[col] < lower_bound) | (features[col] > upper_bound)][col]
            if len(outliers) > 0:
                print(f"  {col}: {len(outliers)} outliers ({len(outliers)/len(features)*100:.1f}%)")
    
    # Categorical features statistics  
    print(f"\n📝 CATEGORICAL FEATURES STATISTICS:")
    if len(categorical_cols) > 0:
        print("Unique values per categorical feature (first 10):")
        for col in categorical_cols[:10]:
            unique_count = features[col].nunique()
            most_common = features[col].mode().iloc[0] if len(features[col].mode()) > 0 else "N/A"
            print(f"  {col}: {unique_count} unique values, most common: '{most_common}'")
    
    # Target variable statistics (if available)
    if label in train_df.columns:
        print(f"\n🎯 TARGET VARIABLE ({label}) STATISTICS:")
        target_stats = train_df[label].describe()
        print(f"Mean: ${target_stats['mean']:,.2f}")
        print(f"Median: ${target_stats['50%']:,.2f}")
        print(f"Std: ${target_stats['std']:,.2f}")
        print(f"Min: ${target_stats['min']:,.2f}")
        print(f"Max: ${target_stats['max']:,.2f}")
        print(f"Skewness: {train_df[label].skew():.3f}")


def print_preprocessed_data_stats(X_processed, y_target=None, original_numeric_cols=None):
    """Print comprehensive statistics for preprocessed data"""
    print("\n" + "="*80)
    print("PREPROCESSED DATA STATISTICS")
    print("="*80)
    
    # Basic info
    print(f"\n📊 PROCESSED DATASET SHAPE:")
    print(f"Features: {X_processed.shape}")
    if y_target is not None:
        print(f"Target: {y_target.shape}")
    
    # Feature composition
    print(f"\n🔧 FEATURE COMPOSITION:")
    print(f"Total features after preprocessing: {X_processed.shape[1]}")
    
    # Identify different types of processed features
    numeric_features = []
    onehot_features = []
    
    for col in X_processed.columns:
        if original_numeric_cols is not None and col in original_numeric_cols:
            numeric_features.append(col)
        else:
            onehot_features.append(col)
    
    print(f"Original numerical features (standardized): {len(numeric_features)}")
    print(f"One-hot encoded features: {len(onehot_features)}")
    
    # Missing values check
    print(f"\n✅ MISSING VALUES CHECK:")
    missing_count = X_processed.isnull().sum().sum()
    print(f"Total missing values: {missing_count}")
    if missing_count == 0:
        print("✅ No missing values - preprocessing successful!")
    else:
        print("❌ Warning: Missing values still present")
        missing_cols = X_processed.isnull().sum()
        missing_cols = missing_cols[missing_cols > 0]
        for col, count in missing_cols.items():
            print(f"  {col}: {count} missing")
    
    # Standardization check for numerical features
    if len(numeric_features) > 0:
        print(f"\n📈 STANDARDIZATION CHECK (Numerical Features):")
        print("Feature statistics after standardization:")
        stats_df = pd.DataFrame({
            'Feature': numeric_features[:10],  # Show first 10
            'Mean': [X_processed[col].mean() for col in numeric_features[:10]],
            'Std': [X_processed[col].std() for col in numeric_features[:10]],
            'Min': [X_processed[col].min() for col in numeric_features[:10]],
            'Max': [X_processed[col].max() for col in numeric_features[:10]]
        })
        
        for _, row in stats_df.iterrows():
            print(f"  {row['Feature'][:20]:20s}: mean={row['Mean']:6.3f}, std={row['Std']:6.3f}, "
                  f"min={row['Min']:6.2f}, max={row['Max']:6.2f}")
        
        # Check if standardization worked
        mean_check = abs(X_processed[numeric_features].mean().mean()) < 0.01
        std_check = abs(X_processed[numeric_features].std().mean() - 1.0) < 0.01
        print(f"\n✅ Standardization validation:")
        print(f"  Mean ≈ 0: {'✅ PASS' if mean_check else '❌ FAIL'}")
        print(f"  Std ≈ 1: {'✅ PASS' if std_check else '❌ FAIL'}")
    
    # One-hot encoding check
    if len(onehot_features) > 0:
        print(f"\n🔢 ONE-HOT ENCODING CHECK:")
        print(f"Number of binary features created: {len(onehot_features)}")
        
        # Check if all one-hot features are binary (0 or 1)
        binary_check = True
        non_binary_features = []
        
        for col in onehot_features[:20]:  # Check first 20 one-hot features
            unique_vals = set(X_processed[col].unique())
            if not unique_vals.issubset({0, 1, 0.0, 1.0}):
                binary_check = False
                non_binary_features.append(col)
        
        if binary_check:
            print("✅ All one-hot features are properly binary (0/1)")
        else:
            print(f"❌ Warning: {len(non_binary_features)} features are not binary")
            for col in non_binary_features[:5]:
                print(f"  {col}: unique values = {X_processed[col].unique()}")
        
        # Show distribution of one-hot features
        print("\nOne-hot feature activation rates (first 10):")
        for col in onehot_features[:10]:
            activation_rate = X_processed[col].mean() * 100
            print(f"  {col[:30]:30s}: {activation_rate:5.1f}% activated")
    
    # Memory usage
    print(f"\n💾 MEMORY USAGE:")
    memory_mb = X_processed.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"Processed features memory usage: {memory_mb:.2f} MB")
    
    # Data types
    print(f"\n📋 DATA TYPES:")
    dtype_counts = X_processed.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} features")


def compare_before_after(train_df, X_processed, y_target=None, test_df=None):
    """Print side-by-side comparison of raw vs preprocessed data"""
    
    # Store original numeric columns for comparison
    label = "SalePrice" 
    features = train_df.drop(columns=['Id', label] if label in train_df.columns else ['Id'])
    original_numeric_cols = features.select_dtypes(include=[np.number]).columns
    
    # Print raw data stats
    print_raw_data_stats(train_df, test_df)
    
    # Print preprocessed data stats
    print_preprocessed_data_stats(X_processed, y_target, original_numeric_cols)
    
    # Summary comparison
    print("\n" + "="*80)
    print("BEFORE vs AFTER SUMMARY")
    print("="*80)
    
    print(f"\n📊 SHAPE COMPARISON:")
    original_shape = train_df.drop(columns=['Id', label] if label in train_df.columns else ['Id']).shape
    print(f"Before: {original_shape[0]} samples × {original_shape[1]} features")
    print(f"After:  {X_processed.shape[0]} samples × {X_processed.shape[1]} features")
    print(f"Feature expansion: {X_processed.shape[1] - original_shape[1]:+d} features")
    
    print(f"\n❌ MISSING VALUES:")
    original_missing = train_df.isnull().sum().sum()
    processed_missing = X_processed.isnull().sum().sum()
    print(f"Before: {original_missing} missing values")
    print(f"After:  {processed_missing} missing values")
    print(f"Improvement: {original_missing - processed_missing} missing values removed")
    
    print(f"\n✅ PREPROCESSING SUCCESS:")
    print("✅ Missing values handled")
    print("✅ Categorical features encoded") 
    print("✅ Numerical features standardized")
    print("✅ Data ready for machine learning!")

def create_parser():
    """
    Create argument parser for MLP training script
    """
    parser = argparse.ArgumentParser(
        description='Train MLP model for house price prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=64, 
                       help='Batch size for training')
    
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of training epochs')
    
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                       help='Learning rate for optimizer')
    
    # Model architecture
    parser.add_argument('--hidden_layers', type=str, default='64,32', 
                       help='Hidden layer sizes separated by commas')
    
    parser.add_argument('--activation_function', type=str, default='relu', 
                       help='Activation function for hidden layers')
    
    parser.add_argument('--dropout_rate', type=float, default=0.2, 
                       help='Dropout rate')
    
    # Optimization
    parser.add_argument('--optimizer', type=str, default='adam', 
                       help='Optimizer to use')
    
    parser.add_argument('--loss_function', type=str, default='mse', 
                       help='Loss function to use')
    
    # Training setup
    parser.add_argument('--validation_split', type=float, default=0.2, 
                       help='Fraction of data to use for validation')
    
    parser.add_argument('--early_stopping', type=str, default='True', 
                       help='Enable early stopping (True/False)')

    parser.add_argument('--log_transform_targets', type=str, default='True', 
                       help='log transform targets (True/False)')
    
    # File paths
    parser.add_argument('--model_save_path', type=str, 
                       default='/home/mila/w/wook/scratch/deep_rl_practice/house_price_mlp/models/house_price_model.h5', 
                       help='Path to save the trained model')
    
    parser.add_argument('--tensorboard_log_dir', type=str, 
                       default='/home/mila/w/wook/scratch/deep_rl_practice/house_price_mlp/logs', 
                       help='Directory for TensorBoard logs')
    
    parser.add_argument('--data_path', type=str, 
                       default='/home/mila/w/wook/scratch/deep_rl_practice/house_price_mlp/house_prices_data', 
                       help='Path to the dataset')
    
    return parser


def create_data_loaders(train_features, train_target, val_features=None, val_target=None, 
                       batch_size=32, shuffle=True):
    """
    Create PyTorch DataLoaders for batch training
    
    Args:
        train_features: Training features (pandas DataFrame or numpy array)
        train_target: Training targets (pandas Series or numpy array)
        val_features: Validation features (optional)
        val_target: Validation targets (optional)
        batch_size: Batch size for training
        shuffle: Whether to shuffle training data
    
    Returns:
        train_loader, val_loader (val_loader is None if no validation data)
    """
    # Convert to tensors
    X_train = torch.tensor(train_features.values if hasattr(train_features, 'values') else train_features, 
                          dtype=torch.float32)
    y_train = torch.tensor(train_target.values if hasattr(train_target, 'values') else train_target, 
                          dtype=torch.float32).view(-1, 1)
    
    # Create training dataset and loader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    
    val_loader = None
    if val_features is not None and val_target is not None:
        X_val = torch.tensor(val_features.values if hasattr(val_features, 'values') else val_features, 
                           dtype=torch.float32)
        y_val = torch.tensor(val_target.values if hasattr(val_target, 'values') else val_target, 
                           dtype=torch.float32).view(-1, 1)
        
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Parse arguments and assign to variables
    parser = create_parser()
    args = parser.parse_args()
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    hidden_layers = [int(x.strip()) for x in args.hidden_layers.split(',')]  # Convert to list of integers
    activation_function = args.activation_function
    dropout_rate = args.dropout_rate
    optimizer = args.optimizer
    loss_function = args.loss_function
    validation_split = args.validation_split
    early_stopping = args.early_stopping.lower() == 'true'  # Convert string to boolean
    log_transform_targets = args.log_transform_targets.lower() == 'true'  # Convert string to boolean
    model_save_path = args.model_save_path
    tensorboard_log_dir = args.tensorboard_log_dir
    data_path = args.data_path
    
    #check if PyTorch Accelerator is available
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    #get data splits and preprocess
    train_raw, test_raw = download_data(data_path)
    train_features, train_target, val_features, val_target, test_features, scalers_dict = preprocess_data(train_raw, test_raw, fit_scalers=True)
    compare_before_after(train_raw, train_features, train_target, test_raw)

    #define model
    mlp_model = StackableMLP(input_dim=train_features.shape[1],
                             hidden_dims=hidden_layers,
                             output_dim=1,
                             activation=activation_function,
                             dropout=dropout_rate,
                             batch_norm=True,
                             output_activation=None,
                             init_method='xavier_uniform')
    mlp_model.to(device)
    print(f"MLP model: {mlp_model}")
 
    #optimizer selection
    if optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(mlp_model.parameters(), lr=learning_rate)
    elif optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(mlp_model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(mlp_model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    #loss function selection
    if loss_function.lower() == 'mse':
        criterion = nn.MSELoss()
    elif loss_function.lower() == 'mae':
        criterion = nn.L1Loss()
    elif loss_function.lower() == 'huber':
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_function}")

    #tensorBoard setup
    writer = SummaryWriter(log_dir=tensorboard_log_dir)   
    print(f"TensorBoard logs will be saved to: {tensorboard_log_dir}")

    #early stopping setup
    early_stopper = EarlyStopper(patience=10, min_delta=0.008) if early_stopping else None
    print(f"Early stopping enabled: {early_stopping}")
    print(f"Training with {len(train_features)} training samples and {len(val_features) if val_features is not None else 0} validation samples")
    print(f"Model will be saved to: {model_save_path}")

    #try log transformation on target variable
    if log_transform_targets:
        train_target = np.log1p(train_target)  # Apply log1p to target variable
        val_target = np.log1p(val_target) if val_target is not None else None

    #create data loaders
    train_loader, val_loader = create_data_loaders(
        train_features, train_target, val_features, val_target, batch_size
    )

    #training loop
    for epoch in range(epochs):
        mlp_model.train()
        epoch_train_loss = 0.0
        num_batches = 0
        
        #batch training
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            #move to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            #forward pass
            outputs = mlp_model(batch_X)
            loss = criterion(outputs, batch_y)
            
            #backward pass and optimization
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            num_batches += 1
        
        #average training loss for the epoch
        avg_train_loss = epoch_train_loss / num_batches
        
        #validation step
        if val_loader is not None:
            mlp_model.eval()
            val_loss = 0.0
            num_val_batches = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    
                    val_outputs = mlp_model(batch_X)
                    if log_transform_targets:
                        val_outputs = torch.expm1(val_outputs)
                        batch_y = torch.expm1(batch_y)

                    loss = criterion(val_outputs, batch_y)
                    val_loss += loss.item()
                    num_val_batches += 1
            
            avg_val_loss = val_loss / num_val_batches
            
            #log to TensorBoard
            if writer:
                writer.add_scalar('Loss/train', avg_train_loss, epoch)
                writer.add_scalar('Loss/val', avg_val_loss, epoch)
            
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            #check early stopping condition
            if early_stopper and early_stopper.early_stop(avg_val_loss):
                print("Early stopping triggered")
                break
        
        else:
            if writer:
                writer.add_scalar('Loss/train', avg_train_loss, epoch)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}")

    #convert test features into pytorch tensors
    test_features_tensor = torch.tensor(test_features.values, dtype=torch.float32).to(device)
    mlp_model.eval()  # Set the model to evaluation mode
    all_predictions = []
    with torch.no_grad():  # Disable gradient calculation for evaluation
        outputs = mlp_model(test_features_tensor)  # Forward pass
        outputs.cpu().numpy()

    #save predictions to a CSV file
    predictions_df = pd.DataFrame({
        'Id': test_raw['Id'],
        'SalePrice': outputs.numpy().flatten()
    })
    predictions_df.to_csv('house_price_predictions.csv', index=False)

    #save the trained model
    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))
    torch.save(mlp_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    

    




    

    

    


