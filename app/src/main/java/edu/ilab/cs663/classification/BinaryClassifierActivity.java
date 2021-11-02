package edu.ilab.cs663.classification;

/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Size;
import android.util.TypedValue;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Toast;

import androidx.annotation.UiThread;

import java.io.IOException;
import java.util.List;

import edu.ilab.cs663.R;
import edu.ilab.cs663.classification.env.BorderedText;
import edu.ilab.cs663.classification.env.Logger;
import edu.ilab.cs663.classification.tflite.KClassifier;
import edu.ilab.cs663.classification.tflite.KClassifier.Device;

public class BinaryClassifierActivity
        extends CameraActivity implements OnImageAvailableListener {

    public static String imageFileURL;

    // flag for pushing records
    int flag = 0;

    private static final Logger LOGGER = new Logger();
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    private static final float TEXT_SIZE_DIP = 10;
    private String modelName;
    private Bitmap rgbFrameBitmap = null;
    private long lastProcessingTimeMs;
    private Integer sensorOrientation;
    private KClassifier KClassifier;
    private BorderedText borderedText;
    /**
     * Input image size of the model along x axis.
     */
    private int imageSizeX;
    /**
     * Input image size of the model along y axis.
     */
    private int imageSizeY;

    //////////////////// STARTUP //////////////////
    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        LOGGER.d("onCreate " + this);
        super.onCreate(null);
        modelName = "baseModel";
        configureModelSpinner();

    }

    ////////////////// VIEW & CAMERA UTILITY ////////////////////////////
    @Override
    @UiThread
    protected void showResultsInBottomSheet(List<KClassifier.Recognition> results) {
        if (results != null && results.size() >= 2) {
            KClassifier.Recognition recognition = results.get(0);
            if (recognition != null) {
                if (recognition.getTitle() != null)
                    recognitionTextView.setText(recognition.getTitle());
                if (recognition.getConfidence() != null)
                    recognitionValueTextView.setText(
                            String.format("%.2f", (100 * recognition.getConfidence())) + "%");
            }

            KClassifier.Recognition recognition1 = results.get(1);
            if (recognition1 != null) {
                if (recognition1.getTitle() != null)
                    recognition1TextView.setText(recognition1.getTitle());
                if (recognition1.getConfidence() != null)
                    recognition1ValueTextView.setText(
                            String.format("%.2f", (100 * recognition1.getConfidence())) + "%");
            }

        }
    }

    @Override
    protected String getModelDirectory() {
        return "dogcat";
    }

    @Override
    protected String getModelPath() {
        return getModelDirectory() + "/" + getModelName() + ".tflite";
    }

    @Override
    protected String getLabelPath() {
        return getModelDirectory() + "/" + "labels.txt";
    }

    @Override
    protected String getModelName() {
        return modelName;
    }


    /**
     * Method for configuring list of models available for selection
     * from an existing array in string.xml resource
     */
    private void configureModelSpinner() {
        String[] models = getResources().getStringArray(R.array.dogcat);
        ArrayAdapter<CharSequence> modelSpinnerAdapter = new ArrayAdapter<>(
                this, android.R.layout.simple_spinner_item, models);
        modelSpinnerAdapter.setDropDownViewResource(
                android.R.layout.simple_spinner_dropdown_item);
        modelSpinner.setAdapter(modelSpinnerAdapter);
        // TODO: fix theme issue
        modelSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                // handle for update
                String modelSelected = parent.getItemAtPosition(position).toString();
                modelName = modelSelected;
                if (parent == modelSpinner) {
                    LOGGER.d("Updating for model spinner: " + modelSelected);
                    Toast.makeText(view.getContext(), modelSelected + " loaded!", Toast.LENGTH_SHORT).show();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) { // Add here for case of no selection
            }
        });
//        setModelDirectory("dogcat");
        LOGGER.d("Starting directory: " + String.valueOf(this.getModelDirectory()));
    }

    @Override
    protected int getLayoutId() {
        return R.layout.tfe_ic_camera_connection_fragment;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP,
                        getResources().getDisplayMetrics()
                );
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        recreateClassifier(getDevice(), getNumThreads(), true);
        if (KClassifier == null) {
            LOGGER.e("No classifier on preview!");
            return;
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    }

    ////////////////////// INFERENCE //////////////////////////////

    /**
     * processImage = method that will process each Image
     * <p>
     * NOTE: This ClassifierActivity extends CameraActivity that  gets the camera input
     * using the functions defined in the file CameraActivity.java.
     * This file depends on AndroidManifest.xml to set the camera orientation.
     * <p>
     * CameraActivity also contains code to capture user preferences from the UI and make
     * them available to other classes via convenience methods.
     * <p>
     * model = Model.valueOf(modelSpinner.getSelectedItem().toString().toUpperCase());
     * device = Device.valueOf(deviceSpinner.getSelectedItem().toString());
     * numThreads = Integer.parseInt(threadsTextView.getText().toString().trim());
     * <p>
     * NOTE 2: The file Classifier.java contains most of the complex logic for processing the
     * camera input and running inference.
     * *
     * * A subclasses of the file exist, in ClassifierFloatMobileNet.java (in other
     * Tensorflowlite examples there is ClassifierQuantizedMobileNet.java),
     * to demonstrate the use of
     * floating point (and quantized) models.
     * <p>
     * The Classifier class implements a static method, create, which is used to instantiate the appropriate subclass based on the supplied model type (quantized vs floating point).
     */
    @Override
    protected void processImage() {
        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth,
                0, 0, previewWidth, previewHeight
        );
        final int cropSize = Math.min(previewWidth, previewHeight);
        flag++;

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        if (KClassifier != null) {
                            final long startTime = SystemClock.uptimeMillis();
                            //cs663:  ADD any OTHER preprocessing here

                            //Calling the Classifier
                            final List<KClassifier.Recognition> results =
                                    KClassifier.recognizeImage(rgbFrameBitmap, sensorOrientation);
                            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                            LOGGER.v("Detect: %s", results);

                            //cs663:  ADD  Any Post Processing (after recognition)  code here
                            //Following code creates  display that shows results
                            runOnUiThread(
                                    new Runnable() {
                                        @Override
                                        public void run() {
                                            //DISPLAY information including recognition results
                                            showResultsInBottomSheet(results);
                                            showFrameInfo(previewWidth + "x" + previewHeight);
                                            showCropInfo(imageSizeX + "x" + imageSizeY);
                                            showCameraResolution(cropSize + "x" + cropSize);
                                            showRotationInfo(String.valueOf(sensorOrientation));
                                            showInference(lastProcessingTimeMs + "ms");
                                            LOGGER.d(String.valueOf(results));
                                        }
                                    });
                        }
                        readyForNextImage();
                    }
                });
    }

    @Override
    protected void onInferenceConfigurationChanged() {
        if (rgbFrameBitmap == null) {
            // Defer creation until we're getting camera frames.
            return;
        }
        final Device device = getDevice();
        final int numThreads = getNumThreads();
        runInBackground(() -> recreateClassifier(device, numThreads, true));
    }

    private void recreateClassifier(Device device, int numThreads, boolean isBinary) {
        if (KClassifier != null) {
            LOGGER.d("Closing classifier.");
            KClassifier.close();
            KClassifier = null;
        }
        try {
            LOGGER.d(
                    "Creating classifier (device=%s, numThreads=%d)", device, numThreads);
            LOGGER.d("Recreating...\nModel Path: " + getModelPath() + "\nLabel Path: " + getLabelPath());
            KClassifier = KClassifier.create(this, device, numThreads, isBinary, getModelPath(), getLabelPath());
        } catch (IOException e) {
            LOGGER.e(e, "Failed to create classifier.");
        }

        // Updates the input image size.
        imageSizeX = KClassifier.getImageSizeX();
        imageSizeY = KClassifier.getImageSizeY();
    }


}
