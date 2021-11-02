/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package edu.ilab.cs663.classification.tflite;

import android.app.Activity;
import android.util.Log;

import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.ops.NormalizeOp;

import java.io.IOException;

/**
 * This TensorFlowLite classifier works with the float MobileNet model.
 */
public class KClassifierFloatMobileNet extends KClassifier {
    public static final String TAG = "KClassifierFloatMobileNet";
    /**
     * Float MobileNet requires additional normalization of the used input.
     */
    private static float IMAGE_MEAN = 127.5f;

    private static float IMAGE_STD = 127.5f;
//    private static float IMAGE_MEAN = 0f;
//
//    private static float IMAGE_STD = 255.0f;

    /**
     * Float model does not need dequantization in the post-processing. Setting mean and std as 0.0f
     * and 1.0f, repectively, to bypass the normalization.
     */
    private static final float PROBABILITY_MEAN = 0.0f;

    private static final float PROBABILITY_STD = 1.0f;
    private String modelPath, labelPath;

    /**
     * Initializes a {@code ClassifierFloatMobileNet}.
     *
     * @param activity
     */
    public KClassifierFloatMobileNet(Activity activity, Device device, int numThreads, boolean isBinary, String modelPath, String labelPath)
            throws IOException {
        super(activity, device, numThreads, isBinary,modelPath, labelPath);
        this.modelPath = modelPath;
        this.labelPath = labelPath;
        if (isBinary) {
            IMAGE_MEAN = 0f;
            IMAGE_STD = 255.0f;
        } else {
            IMAGE_MEAN = 127.5f;
            IMAGE_STD = 127.5f;
        }
        Log.d(TAG, "IMAGE_MEAN:" + String.valueOf(IMAGE_MEAN) + "IMAGE_STD:" + String.valueOf(IMAGE_STD));
    }

    // TODO: Specify model.tflite as the model file and labels.txt as the label file

    @Override
    protected String getModelPath() {
        return this.modelPath;
//        return "flowers/model.tflite";
    }

    @Override
    protected String getLabelPath() {
        return this.labelPath;
//        return "flowers/labels.txt";
    }


    @Override
    protected TensorOperator getPreprocessNormalizeOp() {
        return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
    }

    @Override
    protected TensorOperator getPostprocessNormalizeOp() {
        return new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD);
    }
}
