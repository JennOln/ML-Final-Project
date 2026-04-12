import pandas as pd
import jax
import jax.numpy as jnp

# source jax_env/bin/activate
# mlflow ui
#levantar el servidor
# uvicorn main:app --reload --host 0.0.0.0 --port 8000

class booksData:
    def __init__(self, file):
        self.data = pd.read_csv(file)
        """Separate from data set"""
        self.X = None
        self.Y = None
        self.features = None

    def preprocess_data(self):
        """ 
            Preprocess Data
            Implementar algun ajuste con los autores famosos con los que van comenzando
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

        # famous Autor
        columna_autor = 'author' 
        conteo_autores = self.data[columna_autor].value_counts()
        
        # Filtramos los que tienen 3 o más libros publicados
        autores_famosos = conteo_autores[conteo_autores >= 3].index
        
        # Creamos la nueva columna (1 si es famoso, 0 si no lo es)
        self.data['is_established_author'] = self.data[columna_autor].isin(autores_famosos).astype(float)

        self.data['macro_genre'] = self.data['category_name'].map(mapeo).fillna('Others')
        genre_dummies = pd.get_dummies(self.data['macro_genre'], prefix='genre', dtype=int)
        
        self.data = pd.concat([self.data, genre_dummies], axis=1)
        self.data = self.data.drop(['category_name', 'macro_genre'], axis=1)
        
        print(f"New columns for genre{len(genre_dummies.columns)} columnas macro.")
        print("Columns for genre:")
        print(genre_dummies.columns.tolist())

    def extract_features_target(self):
        """ 
            Feature Engineering
            Fase de Feature Engineering: get matrix X and vector y 
        """ 

        col_dummies = [col for col in self.data.columns if col.startswith('genre_')]
        print(f"Columnas de género (One-Hot): {col_dummies}")

        col_numerical = [
            'stars',
            'price'
            ]
        
        col_boolean = ['is_established_author','isKindleUnlimited', 'isEditorsPick', 'isGoodReadsChoice']
        for col in col_boolean:
            self.data[col] = self.data[col].astype(int)

        self.features = col_numerical + col_dummies + col_boolean

        umbral_elite = self.data['reviews'].quantile(0.75)
        self.data['isElite'] = (self.data['reviews'] >= umbral_elite).astype(int)
        print(f"Umbral de reseñas para ser Élite: {umbral_elite}")


        self.X = self.data[self.features].values
        self.y = self.data['isElite'].values
        return self.X, self.y

    def normalized_data(self):
        """Normalización implemented with JAX"""
        X_jax = jnp.array(self.X, dtype=jnp.float32)
        
        # 2. Partimos la matriz. Las primeras 2 columnas son stars y price
        X_num = X_jax[:, :2] 
        X_bin = X_jax[:, 2:] # El resto se queda intacto (0s y 1s puros)

        # 3. Calculamos mu y sigma SOLO para stars y price
        self.mu_train = jnp.mean(X_num, axis=0)
        self.sigma_train = jnp.std(X_num, axis=0)
        
        X_num_scaled = (X_num - self.mu_train) / (self.sigma_train + 1e-8)
        
        # 4. Volvemos a pegar la matriz
        self.X_scaled = jnp.concatenate([X_num_scaled, X_bin], axis=1)
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
