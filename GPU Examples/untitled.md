{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Week 1 (Template for Keras)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "import os"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(tf.__version__)\n",
    "print(\"Num GPUs Available: \", len(tf.config.experimental.list_physical_devices('GPU')))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(optimizer='sgd', loss='mean_squared_error')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)\n",
    "ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.fit(xs, ys, epochs=500)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(model.predict([10.0]))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Week 2 (Simple DNN) & 3 (CNN)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "#print(os.environ)\n",
    "#os.environ[\"CUDA_VISIBLE_DEVICES\"] = \"-1\"\n",
    "#print(os.environ[\"CUDA_VISIBLE_DEVICES\"])\n",
    "import time"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "print(tf.__version__)\n",
    "print(\"Num GPUs Available: \", len(tf.config.experimental.list_physical_devices('GPU')))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "mnist = tf.keras.datasets.fashion_mnist"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "(training_images, training_labels), (test_images, test_labels) = mnist.load_data()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "np.set_printoptions(linewidth=200)\n",
    "import matplotlib.pyplot as plt\n",
    "plt.imshow(training_images[0])\n",
    "#print(training_labels[0])\n",
    "#print(training_images[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "training_images  = training_images / 255.0\n",
    "test_images = test_images / 255.0\n",
    "\n",
    "training_images_cnn=training_images.reshape(60000, 28, 28, 1)\n",
    "test_images_cnn = test_images.reshape(10000, 28, 28, 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(training_images.shape)\n",
    "print(test_images.shape)\n",
    "print(training_images_cnn.shape)\n",
    "print(test_images_cnn.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "modeldnn = tf.keras.models.Sequential([\n",
    "                                    #tf.keras.layers.Flatten(input_shape=(28,28)), \n",
    "                                    tf.keras.layers.Flatten(), \n",
    "                                    tf.keras.layers.Dense(512, activation=tf.nn.relu), \n",
    "                                    tf.keras.layers.Dense(512, activation=tf.nn.relu), \n",
    "                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])\n",
    "\n",
    "#Why doesn't it error out when last layer doesn't match number of classes?\n",
    "\n",
    "modelcnn = tf.keras.models.Sequential([\n",
    "                                    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),\n",
    "                                    tf.keras.layers.MaxPooling2D(2, 2),\n",
    "                                    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),\n",
    "                                    tf.keras.layers.MaxPooling2D(2,2), \n",
    "                                    tf.keras.layers.Flatten(), \n",
    "                                    tf.keras.layers.Dense(512, activation=tf.nn.relu), \n",
    "                                    tf.keras.layers.Dense(512, activation=tf.nn.relu), \n",
    "                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "class myCallback(tf.keras.callbacks.Callback):\n",
    "  def on_epoch_end(self, epoch, logs={}):\n",
    "    if(logs.get('accuracy')>.99):\n",
    "      print(\"\\nReached 99% accuracy so cancelling training!\")\n",
    "      self.model.stop_training = True\n",
    "callbacks = myCallback()\n",
    "        \n",
    "modeldnn.compile(optimizer = tf.optimizers.Adam(), loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])\n",
    "modelcnn.compile(optimizer = tf.optimizers.Adam(), loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])\n",
    "modeldnn.summary()\n",
    "modelcnn.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "start = time.time()\n",
    "#model.fit(training_images, training_labels, epochs=5, batch_size=128)\n",
    "modeldnn.fit(training_images, training_labels, epochs=20, callbacks=[callbacks])\n",
    "end = time.time()\n",
    "print(end - start)\n",
    "test_loss = modeldnn.evaluate(test_images, test_labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_loss = modeldnn.evaluate(test_images, test_labels)\n",
    "print(test_loss)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "start = time.time()\n",
    "#model.fit(training_images, training_labels, epochs=5, batch_size=128)\n",
    "modelcnn.fit(training_images_cnn, training_labels, epochs=20, callbacks=[callbacks])\n",
    "end = time.time()\n",
    "print(end - start)\n",
    "test_loss = modelcnn.evaluate(test_images_cnn, test_labels)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "classifications = model.predict(test_images)\n",
    "\n",
    "print(classifications[1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(test_labels[:100])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#not working\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "f, axarr = plt.subplots(3,4)\n",
    "FIRST_IMAGE=0\n",
    "SECOND_IMAGE=7\n",
    "THIRD_IMAGE=26\n",
    "CONVOLUTION_NUMBER = 1\n",
    "from tensorflow.keras import models\n",
    "layer_outputs = [layer.output for layer in modelcnn.layers]\n",
    "activation_model = tf.keras.models.Model(inputs = modelcnn.input, outputs = layer_outputs)\n",
    "for x in range(0,4):\n",
    "  f1 = activation_model.predict(test_images_cnn[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]\n",
    "  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')\n",
    "  axarr[0,x].grid(False)\n",
    "  f2 = activation_model.predict(test_images_cnn[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]\n",
    "  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')\n",
    "  axarr[1,x].grid(False)\n",
    "  f3 = activation_model.predict(test_images_cnn[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]\n",
    "  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')\n",
    "  axarr[2,x].grid(False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Week 4 (Image Generator + Validation Loop)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pwd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!wget --no-check-certificate \\\n",
    "    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip \\\n",
    "    -O /tmp/horse-or-human.zip"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!wget --no-check-certificate \\\n",
    "    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip \\\n",
    "    -O /tmp/validation-horse-or-human.zip"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import zipfile\n",
    "\n",
    "local_zip = '/tmp/horse-or-human.zip'\n",
    "zip_ref = zipfile.ZipFile(local_zip, 'r')\n",
    "zip_ref.extractall('/tmp/horse-or-human')\n",
    "local_zip = '/tmp/validation-horse-or-human.zip'\n",
    "zip_ref = zipfile.ZipFile(local_zip, 'r')\n",
    "zip_ref.extractall('/tmp/validation-horse-or-human')\n",
    "zip_ref.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Directory with our training horse pictures\n",
    "train_horse_dir = os.path.join('/tmp/horse-or-human/horses')\n",
    "\n",
    "# Directory with our training human pictures\n",
    "train_human_dir = os.path.join('/tmp/horse-or-human/humans')\n",
    "\n",
    "# Directory with our training horse pictures\n",
    "validation_horse_dir = os.path.join('/tmp/validation-horse-or-human/horses')\n",
    "\n",
    "# Directory with our training human pictures\n",
    "validation_human_dir = os.path.join('/tmp/validation-horse-or-human/humans')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_horse_names = os.listdir(train_horse_dir)\n",
    "print(train_horse_names[:10])\n",
    "\n",
    "train_human_names = os.listdir(train_human_dir)\n",
    "print(train_human_names[:10])\n",
    "\n",
    "validation_horse_hames = os.listdir(validation_horse_dir)\n",
    "print(validation_horse_hames[:10])\n",
    "\n",
    "validation_human_names = os.listdir(validation_human_dir)\n",
    "print(validation_human_names[:10])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print('total training horse images:', len(os.listdir(train_horse_dir)))\n",
    "print('total training human images:', len(os.listdir(train_human_dir)))\n",
    "print('total validation horse images:', len(os.listdir(validation_horse_dir)))\n",
    "print('total validation human images:', len(os.listdir(validation_human_dir)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%matplotlib inline\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.image as mpimg\n",
    "\n",
    "# Parameters for our graph; we'll output images in a 4x4 configuration\n",
    "nrows = 4\n",
    "ncols = 4\n",
    "\n",
    "# Index for iterating over images\n",
    "pic_index = 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Set up matplotlib fig, and size it to fit 4x4 pics\n",
    "fig = plt.gcf()\n",
    "fig.set_size_inches(ncols * 4, nrows * 4)\n",
    "\n",
    "pic_index += 8\n",
    "next_horse_pix = [os.path.join(train_horse_dir, fname) \n",
    "                for fname in train_horse_names[pic_index-8:pic_index]]\n",
    "next_human_pix = [os.path.join(train_human_dir, fname) \n",
    "                for fname in train_human_names[pic_index-8:pic_index]]\n",
    "\n",
    "for i, img_path in enumerate(next_horse_pix+next_human_pix):\n",
    "  # Set up subplot; subplot indices start at 1\n",
    "  sp = plt.subplot(nrows, ncols, i + 1)\n",
    "  sp.axis('Off') # Don't show axes (or gridlines)\n",
    "\n",
    "  img = mpimg.imread(img_path)\n",
    "  plt.imshow(img)\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = tf.keras.models.Sequential([\n",
    "    # Note the input shape is the desired size of the image 300x300 with 3 bytes color\n",
    "    # This is the first convolution\n",
    "    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),\n",
    "    tf.keras.layers.MaxPooling2D(2, 2),\n",
    "    # The second convolution\n",
    "    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),\n",
    "    tf.keras.layers.MaxPooling2D(2,2),\n",
    "    # The third convolution\n",
    "    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),\n",
    "    tf.keras.layers.MaxPooling2D(2,2),\n",
    "    # The fourth convolution\n",
    "    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),\n",
    "    tf.keras.layers.MaxPooling2D(2,2),\n",
    "    # The fifth convolution\n",
    "    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),\n",
    "    tf.keras.layers.MaxPooling2D(2,2),\n",
    "    # Flatten the results to feed into a DNN\n",
    "    tf.keras.layers.Flatten(),\n",
    "    # 512 neuron hidden layer\n",
    "    tf.keras.layers.Dense(512, activation='relu'),\n",
    "    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')\n",
    "    tf.keras.layers.Dense(1, activation='sigmoid')\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.optimizers import RMSprop\n",
    "\n",
    "model.compile(loss='binary_crossentropy',\n",
    "              optimizer=RMSprop(lr=0.001),\n",
    "              metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "\n",
    "# All images will be rescaled by 1./255\n",
    "train_datagen = ImageDataGenerator(rescale=1/255)\n",
    "validation_datagen = ImageDataGenerator(rescale=1/255)\n",
    "\n",
    "# Flow training images in batches of 128 using train_datagen generator\n",
    "train_generator = train_datagen.flow_from_directory(\n",
    "        '/tmp/horse-or-human/',  # This is the source directory for training images\n",
    "        target_size=(300, 300),  # All images will be resized to 150x150\n",
    "        batch_size=128,\n",
    "        # Since we use binary_crossentropy loss, we need binary labels\n",
    "        class_mode='binary')\n",
    "\n",
    "# Flow training images in batches of 128 using train_datagen generator\n",
    "validation_generator = validation_datagen.flow_from_directory(\n",
    "        '/tmp/validation-horse-or-human/',  # This is the source directory for training images\n",
    "        target_size=(300, 300),  # All images will be resized to 150x150\n",
    "        batch_size=32,\n",
    "        # Since we use binary_crossentropy loss, we need binary labels\n",
    "        class_mode='binary')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 138,
   "metadata": {},
   "outputs": [
    {
     "ename": "RuntimeError",
     "evalue": "You must compile your model before training/testing. Use `model.compile(optimizer, loss)`.",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mRuntimeError\u001b[0m                              Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-138-8fe91a69fd3c>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      5\u001b[0m       \u001b[0mverbose\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      6\u001b[0m       \u001b[0mvalidation_data\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mvalidation_generator\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 7\u001b[0;31m       validation_steps=8)\n\u001b[0m",
      "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/training.py\u001b[0m in \u001b[0;36mfit\u001b[0;34m(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_freq, max_queue_size, workers, use_multiprocessing, **kwargs)\u001b[0m\n\u001b[1;32m    794\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mkwargs\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    795\u001b[0m       \u001b[0;32mraise\u001b[0m \u001b[0mTypeError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'Unrecognized keyword arguments: '\u001b[0m \u001b[0;34m+\u001b[0m \u001b[0mstr\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 796\u001b[0;31m     \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_assert_compile_was_called\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    797\u001b[0m     \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_check_call_args\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'fit'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    798\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/training.py\u001b[0m in \u001b[0;36m_assert_compile_was_called\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m   2826\u001b[0m     \u001b[0;31m# (i.e. whether the model is built and its inputs/outputs are set).\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2827\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0moptimizer\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2828\u001b[0;31m       raise RuntimeError('You must compile your model before '\n\u001b[0m\u001b[1;32m   2829\u001b[0m                          \u001b[0;34m'training/testing. '\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2830\u001b[0m                          'Use `model.compile(optimizer, loss)`.')\n",
      "\u001b[0;31mRuntimeError\u001b[0m: You must compile your model before training/testing. Use `model.compile(optimizer, loss)`."
     ]
    }
   ],
   "source": [
    "history = model.fit(\n",
    "      train_generator,\n",
    "      steps_per_epoch=8,  \n",
    "      epochs=15,\n",
    "      verbose=1,\n",
    "      validation_data = validation_generator,\n",
    "      validation_steps=8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import random\n",
    "from tensorflow.keras.preprocessing.image import img_to_array, load_img\n",
    "\n",
    "# Let's define a new Model that will take an image as input, and will output\n",
    "# intermediate representations for all layers in the previous model after\n",
    "# the first.\n",
    "successive_outputs = [layer.output for layer in model.layers[1:]]\n",
    "#visualization_model = Model(img_input, successive_outputs)\n",
    "visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)\n",
    "# Let's prepare a random input image from the training set.\n",
    "horse_img_files = [os.path.join(train_horse_dir, f) for f in train_horse_names]\n",
    "human_img_files = [os.path.join(train_human_dir, f) for f in train_human_names]\n",
    "img_path = random.choice(horse_img_files + human_img_files)\n",
    "\n",
    "img = load_img(img_path, target_size=(300, 300))  # this is a PIL image\n",
    "x = img_to_array(img)  # Numpy array with shape (150, 150, 3)\n",
    "x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)\n",
    "\n",
    "# Rescale by 1/255\n",
    "x /= 255\n",
    "\n",
    "# Let's run our image through our network, thus obtaining all\n",
    "# intermediate representations for this image.\n",
    "successive_feature_maps = visualization_model.predict(x)\n",
    "\n",
    "# These are the names of the layers, so can have them as part of our plot\n",
    "layer_names = [layer.name for layer in model.layers]\n",
    "\n",
    "# Now let's display our representations\n",
    "for layer_name, feature_map in zip(layer_names, successive_feature_maps):\n",
    "  if len(feature_map.shape) == 4:\n",
    "    # Just do this for the conv / maxpool layers, not the fully-connected layers\n",
    "    n_features = feature_map.shape[-1]  # number of features in feature map\n",
    "    # The feature map has shape (1, size, size, n_features)\n",
    "    size = feature_map.shape[1]\n",
    "    # We will tile our images in this matrix\n",
    "    display_grid = np.zeros((size, size * n_features))\n",
    "    for i in range(n_features):\n",
    "      # Postprocess the feature to make it visually palatable\n",
    "      x = feature_map[0, :, :, i]\n",
    "      x -= x.mean()\n",
    "      x /= x.std()\n",
    "      x *= 64\n",
    "      x += 128\n",
    "      x = np.clip(x, 0, 255).astype('uint8')\n",
    "      # We'll tile each filter into this big horizontal grid\n",
    "      display_grid[:, i * size : (i + 1) * size] = x\n",
    "    # Display the grid\n",
    "    scale = 20. / n_features\n",
    "    plt.figure(figsize=(scale * n_features, scale))\n",
    "    plt.title(layer_name)\n",
    "    plt.grid(False)\n",
    "    plt.imshow(display_grid, aspect='auto', cmap='viridis')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
