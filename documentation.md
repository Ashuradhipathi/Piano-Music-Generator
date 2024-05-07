### Recurrent Neural Networks (RNN)

A recurrent neural network is a class of artificial neural networks that make use of sequential information. They are called recurrent because they perform the same function for every single element of a sequence, with the result being dependent on previous computations. Whereas outputs are independent of previous computations in traditional neural networks.

[**Long Short-Term Memory (LSTM)**](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

-   A type of Recurrent Neural Network that can efficiently learn via gradient descent. Using a gating mechanism, LSTMs are able to recognise and encode long-term patterns. LSTMs are extremely useful to solve problems where the network has to remember information for a long period of time as is the case in music and text generation.
-   **LSTM layers** is a Recurrent Neural Net layer that takes a sequence as an input and can return either sequences (return_sequences=True) or a matrix.
-  **Dropout layers** are a regularisation technique that consists of setting a fraction of input units to 0 at each update during the training to prevent overfitting. The fraction is determined by the parameter used with the layer.
- **Dense layers** or **fully connected layers** is a fully connected neural network layer where each input node is connected to each output node.
- **The Activation layer** determines what activation function our neural network will use to calculate the output of a node.


#### Keras

[Keras](https://keras.io/) is a high-level neural networks API that simplifies interactions with [Tensorflow](https://www.tensorflow.org/). It was developed with a focus on enabling fast experimentation.


### Music Data

[Music21](http://web.mit.edu/music21/) is a Python toolkit used for computer-aided musicology. It allows us to teach the fundamentals of music theory, generate music examples and study music.

The data splits into two object types: [Notes](http://web.mit.edu/music21/doc/moduleReference/moduleNote.html#note) and [Chord](http://web.mit.edu/music21/doc/moduleReference/moduleChord.html)s. Note objects contain information about the **pitch**, **octave**, and **offset** of the Note.

- **Pitch** refers to the frequency of the sound, or how high or low it is and is represented with the letters [A, B, C, D, E, F, G], with A being the highest and G being the lowest.
- [**Octave**](http://web.mst.edu/~kosbar/test/ff/fourier/notes_pitchnames.html) refers to which set of pitches you use on a piano.
- **Offset** refers to where the note is located in the piece.

And Chord objects are essentially a container for a set of notes that are played at the same time.

Used to parse MIDI files and extract musical notes from them
1. `glob.glob("/workspaces/codespaces-jupyter/data/midi_songs/*.mid")`: This line uses the `glob` module to find all the MIDI files in the specified directory.
    
2. `midi = converter.parse(file)`: This line uses the `parse` function from the `music21.converter` module to convert the MIDI file into a `music21` stream object. This object allows you to manipulate and analyze the musical data in the MIDI file.
    
3. `parts = instrument.partitionByInstrument(midi)`: This line separates the different instruments in the MIDI file. If the MIDI file contains multiple instruments, this function will return a `Stream` of `Part` objects, each containing the notes for one instrument.
    
4. `if parts: notes_to_parse = parts.parts[0].recurse() else: notes_to_parse = midi.flat.notes`: This block checks if the MIDI file contains multiple instruments. If it does, it selects the first instrument (`parts.parts[0]`) and extracts its notes. If the MIDI file does not contain multiple instruments, it extracts all the notes from the file.
    
5. The final loop iterates over each note in `notes_to_parse`. If the note is an instance of `music21.note.Note`, it extracts the pitch of the note and appends it to the `notes` list. If the note is an instance of `music21.chord.Chord`, it extracts the pitches of all the notes in the chord and appends them to the `notes` list. The pitches are represented as integers, and the `join` function is used to concatenate them into a string with dots in between.
    

The result is a list of strings, where each string represents a note or a chord from the MIDI files. This list can be used for various purposes, such as training a machine learning model to generate music. 


``` python
sequence_length = 100
for i in range(0, len(notes) - sequence_length, 1):

    sequence_in = notes[i : i + sequence_length]

    sequence_out = notes[i + sequence_length]

    network_input.append([note_to_int[char] for char in sequence_in])

    network_output.append(note_to_int[sequence_out])
```


1. `n_vocab = len(set(notes))`: This line calculates the number of unique notes in your dataset.
    
2. `n_patterns = len(network_input)`: This line gets the total number of patterns or sequences in your input data.
    
3. `network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))`: This line reshapes your input into a format suitable for training a LSTM network. The network expects input to be in the shape (number of sequences, length of each sequence, number of features in each sequence). In this case, the number of features is 1 because each sequence contains only one note.
    
4. `network_input = network_input/float(n_vocab)`: This line normalizes the input data by dividing each input by the total number of unique notes. Normalization helps to speed up training and reduce the likelihood of getting stuck in local optima.
    
5. `network_output = tf.keras.utils.to_categorical(network_output)`: This line converts the output data into one-hot encoded format. One-hot encoding is a common way to handle categorical data in neural networks. Each unique category in the data is represented by a binary vector where only one bit is ‘on’ (1) and the rest are ‘off’ (0).

### Model

1. `model = Sequential()`: This line initializes a new Sequential model. A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
    
2. `model.add(LSTM(256, input_shape=(network_input.shape[1],network_input.shape[2]), return_sequences=True))`: This line adds the first layer to the model, which is a Long Short-Term Memory (LSTM) layer with 256 units. The `input_shape` argument specifies the shape of the input data, and `return_sequences=True` means that the LSTM layer will return the full sequence of outputs (one vector per timestep) instead of just the last output.
    
3. `model.add(Dropout(0.3))`: This line adds a Dropout layer to the model, which randomly sets a fraction (0.3 in this case) of the input units to 0 at each update during training time. This helps to prevent overfitting.
       
4. `model.add(Dense(256))`: This line adds a Dense (fully connected) layer with 256 units.
        
5. `model.add(Dense(n_vocab))`: This line adds another Dense layer with `n_vocab` units, where `n_vocab` is the number of unique notes in your dataset.
    
6. `model.add(Activation('softmax'))`: This line adds a softmax activation function to the model. The softmax function is often used in the final layer of a neural network-based classifier. It will convert the outputs of the previous layer into probability values that sum to 1, which can be interpreted as the model’s predicted probabilities for each note.
    
7. `model.compile(loss='categorical_crossentropy', optimizer='rmsprop')`: This line compiles the model with the categorical crossentropy loss function and the RMSprop optimizer. The loss function is used to measure the model’s performance on the training data, and the optimizer is used to update the model’s weights based on the gradients of the loss function.


1. `filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"`: This line defines a string that will be used as the filename for the saved model weights. The filename includes placeholders for the epoch number and the loss value, which will be filled in automatically during training.
    
2. 
```python
    checkpoint = ModelCheckpoint(filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
 ``` 
    This line creates a `ModelCheckpoint` callback. This callback saves the model weights after each epoch. The weights are only saved if the model’s performance on the validation set (`monitor='loss'`) has improved (`save_best_only=True`). The `mode='min'` argument means that improvement is defined as a decrease in loss.
    
3. `callbacks_list = [checkpoint]`: This line creates a list of callbacks to be used during training. In this case, the list only includes the `ModelCheckpoint` callback.
    
4. `model.fit(network_input, network_output, epochs=200, batch_size=64, callbacks=callbacks_list)`: This line trains the model for 200 epochs with a batch size of 64. The `callbacks` argument is set to the list of callbacks defined earlier.


```python
model = Sequential()

model.add(LSTM(256, input_shape=(network_input.shape[1],network_input.shape[2]), return_sequences=True))

model.add(Dropout(0.3))

model.add(LSTM(512, return_sequences=True))

model.add(Dropout(0.3))

model.add(LSTM(256))

model.add(Dense(256))

model.add(Dropout(0.3))

model.add(Dense(n_vocab))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop') 
```

#### A mapping function to decode the output of the network. This function will map from numerical data to categorical data (from integers to notes).

1. `start = np.random.randint(0, len(network_input)-1)`: This line selects a random starting point for the sequence to be input to the model.
    
2. `pattern = network_input[start]`: This line initializes the input sequence (`pattern`) with the sequence at the randomly selected starting point.
    
3. The `for` loop generates 500 notes. For each note:
    
    - `prediction_input = np.reshape(pattern, (1,len(pattern),1))`: The current sequence (`pattern`) is reshaped to the input shape expected by the model.
        
    - `prediction_input = prediction_input/float(n_vocab)`: The input is normalized by dividing by the number of unique notes (`n_vocab`).
        
    - `prediction = model.predict(prediction_input)`: The model makes a prediction based on the input sequence. The output of this prediction is a probability distribution over the possible notes, represented as indices.
        
    - `index = np.argmax(prediction)`: The index of the note with the highest probability is selected.
        
    - `result = int_to_note[index]`: The selected index is mapped back to the corresponding note using the `int_to_note` dictionary. This is the note that the model has generated.
        
    - `prediction_output.append(result)`: The generated note is added to the output sequence.
        
    - `pattern = np.append(pattern, index)`: The index of the generated note is appended to the input sequence (`pattern`).
        
    - `pattern = pattern[1:len(pattern)]`: The first note in the input sequence is removed to keep the sequence length constant.
        

the RNN is generating indices that represent notes, and these indices are then mapped back to actual notes using the `int_to_note` dictionary. The actual music generation is done in the post-processing step where the indices are converted back to notes. 


```python 
from music21 import stream

midi_stream = stream.Stream(output_notes)

midi_stream.write('midi', fp='test_output.mid')
```
