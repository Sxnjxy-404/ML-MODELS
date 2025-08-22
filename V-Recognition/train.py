import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os

def create_dataset():
    """Create the tennis dataset with more samples for better training"""
    # Original dataset
    base_data = {
        'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny',
                    'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
        'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild',
                        'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
        'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High',
                     'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
        'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak',
                 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
        'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No',
                       'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
    }
    
    # Extended dataset with additional logical combinations
    extended_data = {
        'Outlook': ['Sunny', 'Sunny', 'Rain', 'Rain', 'Overcast', 'Overcast', 'Sunny', 'Rain',
                   'Overcast', 'Sunny', 'Rain', 'Overcast', 'Sunny', 'Rain', 'Overcast'],
        'Temperature': ['Cool', 'Mild', 'Hot', 'Cool', 'Hot', 'Mild', 'Hot', 'Mild',
                       'Cool', 'Cool', 'Hot', 'Hot', 'Mild', 'Cool', 'Mild'],
        'Humidity': ['Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'High',
                    'High', 'High', 'Normal', 'High', 'Normal', 'Normal', 'High'],
        'Wind': ['Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Strong', 'Weak', 'Strong',
                'Weak', 'Strong', 'Weak', 'Strong', 'Weak', 'Strong', 'Weak'],
        'PlayTennis': ['Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No',
                      'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes']
    }
    
    # Combine datasets
    all_data = {}
    for key in base_data.keys():
        all_data[key] = base_data[key] + extended_data[key]
    
    return pd.DataFrame(all_data)

def encode_features(df):
    """Encode categorical features and return encoders"""
    label_encoders = {}
    df_encoded = df.copy()
    
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df_encoded[column] = le.fit_transform(df[column])
            label_encoders[column] = le
    
    return df_encoded, label_encoders

def evaluate_models(X_train, X_test, y_train, y_test, label_encoders):
    """Train and evaluate different models"""
    models = {
        'Decision Tree': DecisionTreeClassifier(criterion='entropy', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree (Optimized)': DecisionTreeClassifier(
            criterion='entropy', 
            max_depth=5, 
            min_samples_split=3,
            random_state=42
        )
    }
    
    results = {}
    
    print("🔍 Model Evaluation Results:")
    print("=" * 50)
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=3)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"\n📊 {name}:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   CV Score: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        # Classification report
        print("   Classification Report:")
        target_names = ['No Tennis', 'Play Tennis']
        report = classification_report(y_test, y_pred, target_names=target_names)
        print("   " + report.replace('\n', '\n   '))
    
    return results

def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning for Decision Tree"""
    print("\n🔧 Hyperparameter Tuning...")
    
    param_grid = {
        'criterion': ['entropy', 'gini'],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 3, 5],
        'min_samples_leaf': [1, 2, 3]
    }
    
    dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(dt, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"✅ Best parameters: {grid_search.best_params_}")
    print(f"✅ Best CV score: {grid_search.best_score_:.3f}")
    
    return grid_search.best_estimator_

def visualize_feature_importance(model, feature_names, label_encoders):
    """Visualize feature importance"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        plt.figure(figsize=(10, 6))
        indices = np.argsort(importances)[::-1]
        
        plt.title('Feature Importance')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices])
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        # Save plot
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("📈 Feature importance plot saved as 'feature_importance.png'")
        
        return importances

def create_decision_tree_visualization(model, feature_names, class_names):
    """Create decision tree visualization"""
    try:
        from sklearn.tree import export_text
        
        tree_rules = export_text(model, feature_names=feature_names)
        
        with open('decision_tree_rules.txt', 'w') as f:
            f.write(tree_rules)
        
        print("🌳 Decision tree rules saved as 'decision_tree_rules.txt'")
        
    except ImportError:
        print("⚠️  Graphviz not available for tree visualization")

def save_model_info(model, label_encoders, accuracy, feature_names):
    """Save model information and metadata"""
    model_info = {
        'model_type': type(model).__name__,
        'accuracy': float(accuracy),
        'training_date': datetime.now().isoformat(),
        'feature_names': feature_names,
        'label_mappings': {},
        'model_parameters': model.get_params() if hasattr(model, 'get_params') else {}
    }
    
    # Save label encoder mappings
    for name, encoder in label_encoders.items():
        model_info['label_mappings'][name] = {
            'classes': encoder.classes_.tolist(),
            'mapping': {cls: int(idx) for idx, cls in enumerate(encoder.classes_)}
        }
    
    with open('model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("📋 Model information saved as 'model_info.json'")

def test_model_predictions(model, label_encoders):
    """Test model with sample predictions"""
    print("\n🧪 Testing Model Predictions:")
    print("=" * 40)
    
    test_cases = [
        ['Sunny', 'Hot', 'High', 'Weak'],      # Should be No
        ['Overcast', 'Mild', 'Normal', 'Weak'], # Should be Yes
        ['Rain', 'Cool', 'Normal', 'Weak'],     # Should be Yes
        ['Sunny', 'Cool', 'Normal', 'Strong']   # Should be Yes
    ]
    
    for i, case in enumerate(test_cases, 1):
        # Create DataFrame
        test_df = pd.DataFrame([{
            'Outlook': case[0],
            'Temperature': case[1],
            'Humidity': case[2],
            'Wind': case[3]
        }])
        
        # Encode
        for col in test_df.columns:
            test_df[col] = label_encoders[col].transform(test_df[col])
        
        # Predict
        prediction = model.predict(test_df)[0]
        confidence = max(model.predict_proba(test_df)[0]) if hasattr(model, 'predict_proba') else None
        
        result = "Yes" if prediction == 1 else "No"
        conf_str = f" (Confidence: {confidence:.1%})" if confidence else ""
        
        print(f"Test {i}: {' '.join(case)} → {result}{conf_str}")

def main():
    """Main training function"""
    print("🎾 Tennis Prediction Model Training")
    print("=" * 50)
    
    # Create dataset
    print("📊 Creating dataset...")
    df = create_dataset()
    print(f"✅ Dataset created with {len(df)} samples")
    print(f"📈 Class distribution:")
    print(df['PlayTennis'].value_counts())
    
    # Encode features
    print("\n🔤 Encoding categorical features...")
    df_encoded, label_encoders = encode_features(df)
    
    # Split features and target
    X = df_encoded.drop('PlayTennis', axis=1)
    y = df_encoded['PlayTennis']
    feature_names = X.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"✅ Data split - Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Evaluate different models
    results = evaluate_models(X_train, X_test, y_train, y_test, label_encoders)
    
    # Hyperparameter tuning
    best_model = hyperparameter_tuning(X_train, y_train)
    
    # Train final model on full training set
    print("\n🎯 Training final optimized model...")
    best_model.fit(X_train, y_train)
    
    # Final evaluation
    final_predictions = best_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, final_predictions)
    
    print(f"🏆 Final Model Accuracy: {final_accuracy:.3f}")
    
    # Feature importance
    feature_importance = visualize_feature_importance(
        best_model, feature_names, label_encoders
    )
    
    # Decision tree visualization
    create_decision_tree_visualization(
        best_model, feature_names, ['No Tennis', 'Play Tennis']
    )
    
    # Save models and encoders
    print("\n💾 Saving model and encoders...")
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    # Save model information
    save_model_info(best_model, label_encoders, final_accuracy, feature_names)
    
    # Test predictions
    test_model_predictions(best_model, label_encoders)
    
    print("\n✅ Training completed successfully!")
    print("📁 Files created:")
    print("   - model.pkl (trained model)")
    print("   - label_encoders.pkl (feature encoders)")
    print("   - model_info.json (model metadata)")
    print("   - feature_importance.png (feature importance plot)")
    print("   - decision_tree_rules.txt (decision tree rules)")
    
    return best_model, label_encoders, final_accuracy

if __name__ == '__main__':
    try:
        model, encoders, accuracy = main()
        print(f"\n🎉 Model training successful! Final accuracy: {accuracy:.1%}")
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        raise