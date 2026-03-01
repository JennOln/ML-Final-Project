import pandas as pd
import jax
import jax.numpy as jnp

# source jax_env/bin/activate


class booksData:
    def __init__(self, file):
        self.data = pd.read_csv(file)
        """Separate from data set"""
        self.X = None
        self.Y = None
        self.features = None

    def preprocess_data(self):
        """ 
            Preprocess Environment_State and Opponent_Strategy data using One-Hot Encoding
        """
        genre_dummies = pd.get_dummies(self.data['category_name'], 
                                     prefix='genre', 
                                     dtype=int
                                     )
        
        # Combinar y eliminar originales
        self.data = pd.concat([self.data, genre_dummies], axis=1)
        self.data = self.data.drop(['category_name'], axis=1) 
        print("Preprocessing complete: Variables convert successfully")

        print("New columns for genre:")
        print(genre_dummies.columns.tolist())
        print(f"Total columns after One-Hot: {len(self.data.columns)}")

    def extract_features_target(self):
        """ 
            Feature Engineering
            Fase de Feature Engineering: get matrix X and vector y 
        """ 
        col_genre = [col for col in self.data.columns if col.startswith('genre')]
        col_others = [
            'stars',
            'reviews',
            'price'
            ]
        col_boolean = ['isKindleUnlimited', 'isEditorsPick', 'isGoodReadsChoice']
        for col in col_boolean:
            self.data[col] = self.data[col].astype(int)

        self.features = col_genre + col_others + col_boolean
        target = 'isBestSeller'        
        self.X = self.data[self.features].values
        self.y = self.data[target].values
        return self.X, self.y

    def normalized_data(self):
        """Normalización implemented with JAX"""
        X_numeric = self.X.astype(float) #astype convert True or False in 1.0 or 0.0
        X_jax = jnp.array(X_numeric) # Convertir la matriz X a JAX array

        mean = jnp.mean(X_jax, axis=0) # Calcular mu por col, axis=0 calcula el promedio de cada característica
        std = jnp.std(X_jax, axis=0) #Calcular la desviación estándar (sigma) por columna
        
        self.X_scaled = (X_jax - mean) / (std + 1e-8) ## 4. Aplicar la fórmula: (x - mu) / sigma
        print("Normalización con JAX complete.")
        return self.X_scaled
    
def main():
    trending_books = booksData('kindle_data-v2.csv') 

    print("---Original Data (No processing)---")
    print(trending_books.data.head())
    print("Columns Data processing:")
    print(trending_books.data.columns.tolist())

    trending_books.preprocess_data()
    X, y = trending_books.extract_features_target()
    print(f"Matrix X (features): {X.shape[0]} samples, columns {X.shape[1]}" )
    print(X[:1])
    print(f"Vector y (target): {y.shape[0]} samples")

    X_scaled = trending_books.normalized_data()
    media_final = jnp.mean(X_scaled, axis=0)
    std_final = jnp.std(X_scaled, axis=0)

    print(f"Media tras normalizar (debe ser cercana a 0): {media_final[0:3]}") 
    print(f"Desviación tras normalizar (debe ser 1): {std_final[0:3]}")

if __name__ == "__main__":
    main()
