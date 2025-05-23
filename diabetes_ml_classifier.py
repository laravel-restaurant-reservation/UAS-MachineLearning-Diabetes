import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
import os
from datetime import datetime
import json

class DiabetesClassifier:
    def __init__(self, data_path):
        """
        Inisialisasi classifier dengan path dataset
        """
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.scalers = {
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler(),
            'NoScaler': None
        }
        
        # Buat folder untuk menyimpan hasil
        self.output_dir = self.create_output_directory()
        
    def create_output_directory(self):
        """
        Membuat direktori output dengan format: train_YYYYMMDD_HHMMSS
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_dir = 'training_results'
        
        # Buat direktori utama jika belum ada
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        # Hitung jumlah folder training yang sudah ada
        existing_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        train_number = len(existing_dirs) + 1
        
        # Buat direktori baru
        output_dir = os.path.join(base_dir, f'train{train_number}_{timestamp}')
        os.makedirs(output_dir)
        
        # Buat subdirektori untuk plot
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir)
        
        return output_dir

    def save_report(self, content, filename):
        """
        Menyimpan report ke file dengan format yang lebih rapi
        """
        # Ganti karakter escape \n dengan newline yang sebenarnya
        content = content.replace('\\n', '\n')
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content)

    def format_section(self, title, content, level=1):
        """
        Format section dengan header yang sesuai
        """
        if level == 1:
            separator = "=" * 80
            prefix = "\n"
        else:
            separator = "-" * 60
            prefix = "  "
        
        return f"{prefix}{title}\n{separator}\n{content}\n"

    def load_data(self):
        """
        Memuat dan menampilkan informasi awal dataset
        """
        print("1. Loading Dataset...")
        self.df = pd.read_csv(self.data_path)
        
        # Buat report dataset
        report = "LAPORAN ANALISIS DATASET DIABETES\\n"
        report += "=" * 50 + "\\n\\n"
        
        report += "1. INFORMASI DATASET\\n"
        report += "-" * 20 + "\\n"
        report += f"Jumlah sampel: {self.df.shape[0]}\\n"
        report += f"Jumlah fitur: {self.df.shape[1]-1}\\n\\n"
        
        report += "2. STATISTIK DESKRIPTIF\\n"
        report += "-" * 20 + "\\n"
        report += str(self.df.describe()) + "\\n\\n"
        
        report += "3. INFORMASI TIPE DATA\\n"
        report += "-" * 20 + "\\n"
        report += str(self.df.dtypes) + "\\n"
        
        self.save_report(report, 'dataset_report.txt')
        return self

    def analyze_data(self):
        """
        Melakukan analisis eksploratori data
        """
        print("\\n2. Exploratory Data Analysis...")
        plots_dir = os.path.join(self.output_dir, 'plots')
        
        # Distribusi kelas
        plt.figure(figsize=(8, 6))
        sns.countplot(data=self.df, x='Outcome')
        plt.title('Distribusi Kelas Target (Diabetes)')
        plt.savefig(os.path.join(plots_dir, 'class_distribution.png'))
        plt.close()

        # Korelasi
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Matriks Korelasi')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'correlation_matrix.png'))
        plt.close()

        # Box plots
        plt.figure(figsize=(15, 10))
        self.df.boxplot()
        plt.title('Distribusi Fitur')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'feature_distribution.png'))
        plt.close()

        # Buat report EDA
        report = "LAPORAN EXPLORATORY DATA ANALYSIS\\n"
        report += "=" * 50 + "\\n\\n"
        
        report += "1. DISTRIBUSI KELAS\\n"
        report += "-" * 20 + "\\n"
        report += str(self.df['Outcome'].value_counts()) + "\\n\\n"
        
        report += "2. KORELASI DENGAN TARGET\\n"
        report += "-" * 20 + "\\n"
        correlations = self.df.corr()['Outcome'].sort_values(ascending=False)
        report += str(correlations) + "\\n"
        
        self.save_report(report, 'eda_report.txt')
        return self

    def preprocess_data(self):
        """
        Melakukan preprocessing data tanpa scaling
        """
        print("\n3. Data Preprocessing...")
        
        # Report preprocessing
        report = self.format_section("LAPORAN PREPROCESSING DATA", "")
        
        # Missing values
        missing_values = self.df.isnull().sum()
        missing_report = "Jumlah Missing Values per Kolom:\n\n"
        for column, count in missing_values.items():
            missing_report += f"{column:20}: {count:5d}\n"
        report += self.format_section("Missing Values", missing_report, 2)
        
        # Outliers
        Q1 = self.df.quantile(0.25)
        Q3 = self.df.quantile(0.75)
        IQR = Q3 - Q1
        
        outliers_report = "Jumlah Outliers per Kolom:\n\n"
        for column in self.df.columns:
            outliers = self.df[(self.df[column] < Q1[column] - 1.5 * IQR[column]) | 
                              (self.df[column] > Q3[column] + 1.5 * IQR[column])][column]
            outliers_report += f"{column:20}: {len(outliers):5d} outliers\n"
        report += self.format_section("Outliers (Sebelum Penanganan)", outliers_report, 2)
        
        # Filter outliers
        self.df = self.df[~((self.df < (Q1 - 1.5 * IQR)) | 
                           (self.df > (Q3 + 1.5 * IQR))).any(axis=1)]
        
        dataset_info = (
            f"Ukuran Dataset:\n"
            f"  - Jumlah sampel: {self.df.shape[0]}\n"
            f"  - Jumlah fitur: {self.df.shape[1]-1}\n"
        )
        report += self.format_section("Informasi Dataset Setelah Preprocessing", dataset_info, 2)
        
        self.save_report(report, 'preprocessing_report.txt')
        
        # Memisahkan fitur dan target
        self.X = self.df.drop('Outcome', axis=1)
        self.y = self.df['Outcome']
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        return self

    def apply_scaling_and_smote(self, X_train, X_test, scaler_name, scaler):
        """
        Menerapkan scaling dan SMOTE pada data
        """
        if scaler is not None:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()
        
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, self.y_train)
        
        return X_train_resampled, X_test_scaled, y_train_resampled

    def train_models(self):
        """
        Melatih berbagai model machine learning dengan berbagai scaling
        """
        print("\n4. Model Training dengan Berbagai Normalisasi...")
        plots_dir = os.path.join(self.output_dir, 'plots')
        
        # Inisialisasi model
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                solver='lbfgs',
                n_jobs=-1
            ),
            'SVM': SVC(kernel='rbf', random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        
        # Report training
        report = self.format_section("LAPORAN TRAINING MODEL", "")
        
        # Untuk setiap scaler
        for scaler_name, scaler in self.scalers.items():
            scaler_results = f"\nHasil Training dengan {scaler_name}:\n\n"
            
            X_train_scaled, X_test_scaled, y_train_resampled = self.apply_scaling_and_smote(
                self.X_train, self.X_test, scaler_name, scaler
            )
            
            # Untuk setiap model
            for name, model in self.models.items():
                model_results = f"Model: {name}\n{'-' * (len(name) + 7)}\n"
                model_clone = clone(model)
                model_clone.fit(X_train_scaled, y_train_resampled)
                
                y_pred = model_clone.predict(X_test_scaled)
                
                accuracy = accuracy_score(self.y_test, y_pred)
                cv_scores = cross_val_score(model_clone, X_train_scaled, y_train_resampled, cv=5)
                conf_matrix = confusion_matrix(self.y_test, y_pred)
                
                result_key = f"{name} ({scaler_name})"
                self.results[result_key] = {
                    'accuracy': accuracy,
                    'cv_scores': cv_scores,
                    'confusion_matrix': conf_matrix,
                    'classification_report': classification_report(self.y_test, y_pred)
                }
                
                # Format hasil untuk report
                model_results += (
                    f"Accuracy: {accuracy:.4f}\n"
                    f"Cross-validation Score: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})\n\n"
                    f"Classification Report:\n{classification_report(self.y_test, y_pred)}\n"
                    f"{'=' * 40}\n"
                )
                scaler_results += model_results
            
            report += self.format_section(f"Hasil dengan {scaler_name}", scaler_results, 2)
            
            # Plot confusion matrix
            for name, model in self.models.items():
                result_key = f"{name} ({scaler_name})"
                conf_matrix = self.results[result_key]['confusion_matrix']
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix - {result_key}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.savefig(os.path.join(plots_dir, f'confusion_matrix_{name.replace(" ", "_").lower()}_{scaler_name.lower()}.png'))
                plt.close()
        
        self.save_report(report, 'training_report.txt')
        return self

    def show_results(self):
        """
        Menampilkan hasil evaluasi model
        """
        print("\n5. Model Evaluation Results...")
        plots_dir = os.path.join(self.output_dir, 'plots')
        
        # Membandingkan akurasi
        accuracies = {name: results['accuracy'] 
                     for name, results in self.results.items()}
        
        # Plot perbandingan akurasi
        plt.figure(figsize=(15, 8))
        model_names = list(self.models.keys())
        scaler_names = list(self.scalers.keys())
        x = np.arange(len(model_names))
        width = 0.2
        
        for i, scaler_name in enumerate(scaler_names):
            accuracies_scaler = [accuracies[f"{model} ({scaler_name})"] 
                               for model in model_names]
            plt.bar(x + i*width, accuracies_scaler, width, label=scaler_name)
        
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy Comparison with Different Scalers')
        plt.xticks(x + width*1.5, model_names, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'model_comparison_with_scalers.png'))
        plt.close()
        
        # Buat laporan hasil akhir
        report = self.format_section("LAPORAN HASIL AKHIR EVALUASI MODEL", "")
        
        # Tabel perbandingan
        comparison_table = "Perbandingan Akurasi Model dengan Berbagai Scaler:\n\n"
        comparison_table += f"{'Model':25} {'Accuracy':10} {'CV Score':15} {'Std Dev':10}\n"
        comparison_table += "-" * 65 + "\n"
        
        for scaler_name in scaler_names:
            comparison_table += f"\n{scaler_name}:\n"
            comparison_table += "-" * 20 + "\n"
            for model_name in model_names:
                key = f"{model_name} ({scaler_name})"
                acc = accuracies[key]
                cv_mean = self.results[key]['cv_scores'].mean()
                cv_std = self.results[key]['cv_scores'].std()
                comparison_table += f"{model_name:25} {acc:.4f}     {cv_mean:.4f}      ±{cv_std*2:.4f}\n"
        
        report += self.format_section("Perbandingan Model", comparison_table, 2)
        
        # Model terbaik
        best_combination = max(accuracies.items(), key=lambda x: x[1])
        best_model_info = (
            f"Model            : {best_combination[0]}\n"
            f"Accuracy         : {best_combination[1]:.4f}\n"
            f"Cross-validation : {self.results[best_combination[0]]['cv_scores'].mean():.4f} "
            f"(±{self.results[best_combination[0]]['cv_scores'].std()*2:.4f})\n\n"
            f"Classification Report:\n"
            f"{self.results[best_combination[0]]['classification_report']}"
        )
        report += self.format_section("Model Terbaik", best_model_info, 2)
        
        self.save_report(report, 'final_report.txt')
        
        # Simpan hasil dalam format JSON
        results_dict = {
            'accuracies': accuracies,
            'best_model': {
                'name': best_combination[0],
                'accuracy': float(best_combination[1])
            }
        }
        
        with open(os.path.join(self.output_dir, 'results.json'), 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        return self

def main():
    classifier = DiabetesClassifier('diabetes_Dataset_cleaned.csv')
    
    (classifier
     .load_data()
     .analyze_data()
     .preprocess_data()
     .train_models()
     .show_results())

if __name__ == "__main__":
    main() 