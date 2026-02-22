import pandas as pd
import jax
import jax.numpy as jnp

# source jax_env/bin/activate


class behaviorData:
    def __init__(self, file='npc_behavior'):
        self.data = pd.read_csv(file)
        """Separate from data set"""
        self.X = None
        self.Y = None
        self.features = None

    def preprocess_data(self):
        """ preprocess Environment_State and Opponent_Strategy data using One-Hot Encoding"""
        env_dummies = pd.get_dummies(self.data['Environment_State'], 
                                     prefix='Env', 
                                     )
        opp_dummies = pd.get_dummies(self.data['Opponent_Strategy'], 
                                     prefix='Opp')
        
        # Combinar y eliminar originales
        self.data = pd.concat([self.data, env_dummies, opp_dummies], axis=1)
        self.data = self.data.drop(['Environment_State', 'Opponent_Strategy'], axis=1)
        print("Preprocessing complete: Variables convert successfully")

    def extract_features_target(self):
        """ Fase de Feature Engineering: get matrix X and vector y """ 
        self.features = [
            'Env_Calm',
            'Env_Chaotic',
            'Env_Combat',
            'Env_Stealth',
            'Opp_Balanced',
            'Opp_Defensive',
            'Opp_Random',
            'Sensory_Input_Level',
            'Decision_Time', 
            'Policy_Confidence']
        target = 'NPC_Action_Type'
        
        self.X = self.data[self.features].values
        self.y = self.data[target].values
        return self.X, self.y

    def normalized_data(self):
        """Normalización Z-score implemented with JAX"""
        X_jax = jnp.array(self.X) # Convertir la matriz X a JAX array
        mean = jnp.mean(X_jax, axis=0) # Calcular mu por col, axis=0 calcula el promedio de cada característica
        std = jnp.std(X_jax, axis=0) #Calcular la desviación estándar (sigma) por columna
        self.X_scaled = (X_jax - mean) / (std + 1e-8) ## 4. Aplicar la fórmula: (x - mu) / sigma
        print("Normalización con JAX complete.")
        return self.X_scaled
    
def main():
    # 1. Instanciar
    npc_behavior = behaviorData('npc_behavior.csv') 

    print("--- Original Data (No processing)---")
    print(npc_behavior.data.head())
    print("Columns Data processing:")
    print(npc_behavior.data.columns.tolist())

    npc_behavior.preprocess_data()
    X, y = npc_behavior.extract_features_target()
    print(f"Matrix X (features): {X.shape[0]} samples, columns {X.shape[1]}" )
    print(X[:1])
    print(f"Vector y (target): {y.shape[0]} samples")

    X_scaled = npc_behavior.normalized_data()
    media_final = jnp.mean(X_scaled, axis=0)
    std_final = jnp.std(X_scaled, axis=0)

    print(f"Media tras normalizar (debe ser cercana a 0): {media_final[0:3]}") 
    print(f"Desviación tras normalizar (debe ser 1): {std_final[0:3]}")
    

if __name__ == "__main__":
    main()
