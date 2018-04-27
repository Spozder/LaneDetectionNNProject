import matplotlib.image as mpimg
import pickle
import tensorflow as tf

train_images_raw = pickle.load(open("bridge.p", "rb" ))

classifier = tf.estimator.Estimator(
        model_fn= cnn_model,
        model_dir= "./tmp/lane_prediction")

I = train_images_raw
    
plt.figure()
plt.imshow(I[0])

plt.figure()
plt.imshow(I[1])


predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x=np.array(I),
    num_epochs=1,
    shuffle=False)

predictions = list(classifier.predict(input_fn=predict_input_fn))
cnn_road_labels = np.array([p['predict_lane'] for p in predictions])
pickle.dump(cnn_road_labels,open('bridge_CNN_labels.p', "wb" ))
print('Done')