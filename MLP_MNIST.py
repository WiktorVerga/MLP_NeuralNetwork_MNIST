import numpy as np                  #gestisce array numerici
import matplotlib.pyplot as plt     #crea grafici
import seaborn as sns               #crea grafici
import tensorflow as tf             #serve per costruire e addestrare la rete neurale
from tensorflow import keras
from sklearn.metrics import confusion_matrix    #per valutare gli errori del modello
from matplotlib.colors import LinearSegmentedColormap # Necessario per la mappa colori personalizzata


# --- CALLBACK PER LA VISUALIZZAZIONE DEI PESI ---
class WeightVisualizer(keras.callbacks.Callback):
    """Callback per visualizzare l'evoluzione dei pesi del primo strato 'Dense'."""
    def __init__(self, num_neurons_to_show=36, cmap_name='RbG'):
        super().__init__()
        self.num_neurons = num_neurons_to_show
        
        # Definiamo la mappa colori personalizzata: Rosso (-), Nero (0), Verde (+)
        if cmap_name == 'RbG':
            colors = [(1, 0, 0), (0, 0, 0), (0, 1, 0)]  # Rosso, Nero, Verde (RGB)
            self.custom_cmap = LinearSegmentedColormap.from_list("RbG_cmap", colors, N=256)
        else:
            # Fallback robusto se la custom map non funziona bene (es. 'seismic' o 'RdYlGn')
            self.custom_cmap = 'seismic' 

    def on_epoch_end(self, epoch, logs=None):
        # Assumiamo che il primo strato Dense sia all'indice 1 (dopo Flatten)
        # Otteniamo i pesi W1 (Input -> Hidden). [0] prende la matrice dei pesi, [1] i bias.
        weights = self.model.layers[1].get_weights()[0]
        
        # --- PLOT DEI PESI ---
        
        # Calcola il massimo valore assoluto dei pesi per centrare la mappa colori a zero
        v_max = np.max(np.abs(weights))

        plt.figure(figsize=(12, 12))
        plt.suptitle(f"Evoluzione Pesi del Neurone (Epoca {epoch+1})", fontsize=14)

        for i in range(self.num_neurons):
            # Seleziona il vettore dei pesi per il neurone i-esimo
            weights_neuron = weights[:, i] 
            # Rimodella il vettore in una griglia 28x28
            weights_image = weights_neuron.reshape(28, 28)
            
            plt.subplot(8, 16, i + 1)
            
            # Mostra la "maschera" del neurone
            plt.imshow(weights_image, cmap=self.custom_cmap, vmin=-v_max, vmax=v_max)
            
            plt.title(f"Neurone {i+1}", fontsize=8)
            plt.axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show() # Mostra il grafico per l'epoca corrente


def main():
    # --- 1. CARICAMENTO E PREPARAZIONE DATI ---
    print("Caricamento dataset MNIST...")
    # Il dataset è composto da 60.000 immagini di training e 10.000 di test e tutte le immagini sono 28×28 pixel, in scala di grigi
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalizzazione: dividiamo valori dei pixel dividendo per 255 → ottenendo valori tra 0 e 1.
    # Questo aiuta la rete neurale a convergere (imparare) più velocemente.
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    print(f"Dati caricati. Esempio dimensione immagine: {x_train[0].shape}")

    # --- 2. COSTRUZIONE DEL MODELLO (LA RETE NEURALE) ---
    # Usiamo un modello "Sequenziale": i dati passano da un layer all'altro.
    model = keras.Sequential([
        # Layer 1: Appiattisce l'immagine 28x28 in un vettore di 784 numeri
        keras.layers.Flatten(input_shape=(28, 28)),
        
        # Layer 2 (Hidden): 128 neuroni. 'relu' è la funzione di attivazione (f(x)=max(0,x))
        # Serve a far imparare pattern complessi.
        # (simile a come un neurone biologico decide se "sparare" o no).
        keras.layers.Dense(128, activation='relu'),
        
        # Layer 3 (Output): 10 neuroni (uno per ogni cifra da 0 a 9).
        # 'softmax' trasforma i risultati in percentuali di probabilità.
        keras.layers.Dense(10, activation='softmax')
    ])

    # Compilazione del modello
    #Definisci come la rete impara:
    #Adam: ottimizzatore intelligente che aggiorna i pesi
    #cross-entropy: misura l’errore di classificazione
    #accuracy: metrica per valutare quanto indovina
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # --- 3. ADDESTRAMENTO ---
    print("\nInizio addestramento...")

     # Inizializza il callback
    visualizer_callback = WeightVisualizer(num_neurons_to_show=128) 

    # epochs=5 significa che la rete vedrà tutto il dataset per 5 volte.
    # Passa il callback alla funzione fit
    history = model.fit(x_train, y_train, epochs=5, 
                        validation_data=(x_test, y_test),
                        callbacks=[visualizer_callback]) # QUI È LA NOVITÀ!

    # --- 4. ANALISI DEI RISULTATI ---

    # Otteniamo le predizioni su tutto il dataset di test
    predictions = model.predict(x_test)
    # Convertiamo le probabilità in numeri interi (es. [0.1, 0.9, 0.0] -> 1)
    y_pred = np.argmax(predictions, axis=1)

    # A. GRAFICO ACCURATEZZA
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Test Accuracy')
    plt.title('Curva di Apprendimento')
    plt.xlabel('Epoca')
    plt.ylabel('Accuratezza')
    plt.legend()

    # B. MATRICE DI CONFUSIONE
    plt.subplot(1, 2, 2)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice di Confusione')
    plt.xlabel('Predizione della Rete')
    plt.ylabel('Numero Reale')
    plt.tight_layout()
    plt.show()

    # --- 5. GALLERIA DEGLI ERRORI ---
    # Troviamo gli indici dove la rete ha sbagliato
    errori_idx = (y_pred != y_test)
    x_errori = x_test[errori_idx]
    y_errori_reali = y_test[errori_idx]
    y_errori_pred = y_pred[errori_idx]

    print(f"\nNumero totale di errori su 10.000 immagini: {len(x_errori)}")

    # Mostriamo i primi 10 errori
    plt.figure(figsize=(15, 3))
    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.imshow(x_errori[i], cmap='gray')
        plt.title(f"Reale: {y_errori_reali[i]}\nPred: {y_errori_pred[i]}", color='red')
        plt.axis('off')
    plt.suptitle("Cosa confonde la rete? (Analisi degli errori)", fontsize=16)
    plt.show()

if __name__ == "__main__":
    main()