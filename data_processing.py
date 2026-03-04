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
    
        mapeo = {
            'Science Fiction & Fantasy': 'Science Fiction & Fantasy',
            'Literature & Fiction': 'Science Fiction & Fantasy',
            
            'Computers & Technology': 'Education',
            'Engineering & Transportation': 'Education',
            'Science & Math': 'Education',
            'Medical': 'Education',
            'Education & Teaching': 'Education',
            'Politics & Social Sciences': 'Education',
            'Law': 'Education',
            'Business & Money': 'Education',
            
            'Health, Fitness & Dieting': 'Health',
            'Self-Help': 'Health',
            'Parenting & Relationships': 'Health',
            
            'Biographies & Memoirs': 'Culture',
            'History': 'Culture',
            'Arts & Photography': 'Culture',
            'Reference': 'Culture',
            'Foreign Language': 'Culture',
            'Travel': 'Culture',
            'Religion & Spirituality': 'Culture',
            'Sports & Outdoors': 'Culture',
            'Crafts, Hobbies & Home': 'Culture',
            'Cookbooks, Food & Wine': 'Culture',

            'Nonfiction': 'Nonfiction',
            
            "Children's eBooks": 'childs',

            'Teen & Young Adult': 'Teen & Young Adult',
            
            'Comics': 'Entertainment',
            'Humor & Entertainment': 'Entertainment',
            
            'Mystery, Thriller & Suspense': 'Thriller',
            
            'Romance': 'Romance',
            'LGBTQ+ eBooks': 'Romance',
        }
        self.data['macro_genre'] = self.data['category_name'].map(mapeo).fillna('Others')
        genre_dummies = pd.get_dummies(self.data['macro_genre'], prefix='genre', dtype=int)
        
        self.data = pd.concat([self.data, genre_dummies], axis=1)
        self.data = self.data.drop(['category_name', 'macro_genre'], axis=1)
        
        print(f"Preprocesamiento con nuevos grupos completado: {len(genre_dummies.columns)} columnas macro.")

        print("Preprocessing complete: Variables convert successfully")

        print("Columns for genre:")
        print(genre_dummies.columns.tolist())


        #print(f"Total columns after One-Hot: {len(self.data.columns)}")

    def extract_features_target(self):
        """ 
            Feature Engineering
            Fase de Feature Engineering: get matrix X and vector y 
        """ 

        col_dummies = [col for col in self.data.columns if col.startswith('genre_')]
        print(f"Columnas de género (One-Hot): {col_dummies}")

        col_numerical = [
            'stars',
            'reviews',
            'price'
            ]
        
        col_boolean = ['isKindleUnlimited', 'isEditorsPick', 'isGoodReadsChoice']
        for col in col_boolean:
            self.data[col] = self.data[col].astype(int)

        self.features = col_numerical + col_dummies + col_boolean
        target = 'isBestSeller'        

        self.X = self.data[self.features].values
        self.y = self.data[target].values
        return self.X, self.y

    def normalized_data(self):
        """Normalización implemented with JAX"""
        #59.61% de accuracy
        X_numeric = self.X.astype(float) #astype convert True or False in 1.0 or 0.0
        X_jax = jnp.array(X_numeric) # Convertir la matriz X a JAX array

        mean = jnp.mean(X_jax, axis=0) # Calcular mu por col, axis=0 calcula el promedio de cada característica
        std = jnp.std(X_jax, axis=0) #Calcular la desviación estándar (sigma) por columna
        
        self.X_scaled = (X_jax - mean) / (std + 1e-8) ## 4. Aplicar la fórmula: (x - mu) / sigma
        print("Normalización con JAX complete.")
        
        """
        #47% de accuracy
        X_jax = jnp.array(self.X, dtype=jnp.float32)
        X_numeric = X_jax[:, :3] # stars, reviews, price
        X_binarias = X_jax[:, 3:] #genre and booleans

        mu = jnp.mean(X_numeric, axis=0)    # Normalización SOLO para las numéricas
        sigma = jnp.std(X_numeric, axis=0)
        X_numeric_scaled = (X_numeric - mu) / (sigma + 1e-8) # El 1e-8 evita la división por cero si una columna es constante
        
        
        self.X_scaled = jnp.concatenate([X_numeric_scaled, X_binarias], axis=1)
        """
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
