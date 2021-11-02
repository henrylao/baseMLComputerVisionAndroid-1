package edu.ilab.cs663.classification.tflite;//package edu.ilab.cs663.classification.tflite;
//
//
//
//
//import android.app.Activity;
//        import android.graphics.Bitmap;
//        import android.graphics.Color;
//        import android.os.Trace;
//        import android.util.Log;
//
//        import androidx.annotation.NonNull;
//
//        import com.lindronics.flirapp.camera.AffineTransformer;
//        import com.lindronics.flirapp.camera.FrameDataHolder;
//
//        import org.tensorflow.lite.DataType;
//        import org.tensorflow.lite.Interpreter;
//        import org.tensorflow.lite.gpu.GpuDelegate;
//        import org.tensorflow.lite.nnapi.NnApiDelegate;
//        import org.tensorflow.lite.support.common.FileUtil;
//        import org.tensorflow.lite.support.label.TensorLabel;
//        import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
//
//        import java.io.IOException;
//        import java.nio.MappedByteBuffer;
//        import java.util.ArrayList;
//        import java.util.List;
//        import java.util.Locale;
//        import java.util.Map;
//        import java.util.PriorityQueue;
//
//public class ModelHandler {
//
//    /**
//     * Interpreter for inference
//     */
//    private Interpreter tflite;
//
//    /**
//     * Loaded TF model
//     */
//    private MappedByteBuffer tfliteModel;
//
//    /**
//     * Whether this is a binary or multi-class model
//     */
//    private Boolean isBinaryClassifier;
//
//    /**
//     * Labels corresponding to the output of the model.
//     */
//    private List<String> labels;
//
//    /**
//     * Optional GPU delegate for hardware acceleration.
//     */
//    private GpuDelegate gpuDelegate = null;
//
//    /**
//     * Optional NNAPI delegate for hardware acceleration.
//     */
//    private NnApiDelegate nnApiDelegate = null;
//
//    /**
//     * Output probability TensorBuffer.
//     */
//    private TensorBuffer outputProbabilityBuffer;
//
//    /**
//     * Image size along the x axis.
//     */
//    private int imageHeight;
//
//    /**
//     * Image size along the y axis.
//     */
//    private int imageWidth;
//
//    /**
//     * Number of results to show in the UI.
//     */
//    private static final int MAX_RESULTS = 3;
//
//    private AffineTransformer transformer;
//
//    /**
//     * Possible devices to run the model on
//     */
//    public enum Device {
//        CPU,
//        NNAPI,
//        GPU
//    }
//
//    public ModelHandler(Activity activity, Device device, int numThreads, Boolean isBinaryClassifier) throws IOException {
//
//        this.isBinaryClassifier = isBinaryClassifier;
//
//        // Load model
//        tfliteModel = FileUtil.loadMappedFile(activity, "model.tflite");
//
//        // Select device
//        Interpreter.Options tfliteOptions = new Interpreter.Options();
//        switch (device) {
//            case NNAPI:
//                nnApiDelegate = new NnApiDelegate();
//                tfliteOptions.addDelegate(nnApiDelegate);
//                break;
//            case GPU:
//                gpuDelegate = new GpuDelegate();
//                tfliteOptions.addDelegate(gpuDelegate);
//                break;
//            case CPU:
//                break;
//        }
//        tfliteOptions.setNumThreads(numThreads);
//        tflite = new Interpreter(tfliteModel, tfliteOptions);
//
//        // Load labels out from the label file.
//        labels = FileUtil.loadLabels(activity, "labels.txt");
//
//        // Read type and shape of input and output tensors, respectively.
//        int imageTensorIndex = 0;
//        int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape();
//        imageHeight = imageShape[1];
//        imageWidth = imageShape[2];
//
//        int probabilityTensorIndex = 0;
//        int[] probabilityShape =
//                tflite.getOutputTensor(probabilityTensorIndex).shape();
//        DataType probabilityDataType = tflite.getOutputTensor(probabilityTensorIndex).dataType();
//
//        // Create the output tensor and its processor.
//        outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);
//
//        transformer = new AffineTransformer(activity);
//    }
//
//
//    /**
//     * Runs inference and returns the classification results.
//     */
//    public List<Recognition> recognizeImage(final FrameDataHolder images) {
//        Trace.beginSection("recognizeImage");
//
//        // Load receiveImages
//        Trace.beginSection("loadImage");
//        float[][][][] inputImageBuffer = loadImage(images);
//        Trace.endSection();
//
//        // Runs the inference call.
//        Trace.beginSection("runInference");
//        tflite.run(inputImageBuffer, outputProbabilityBuffer.getBuffer().rewind());
//        Trace.endSection();
//
//        if (isBinaryClassifier) {
//            float[] probability = outputProbabilityBuffer.getFloatArray();
//            Trace.endSection();
//
//            // In binary classification, return positive and negative class
//            ArrayList<Recognition> predictions = new ArrayList<>();
//            predictions.add(new Recognition("0", labels.get(0), 1 - probability[0]));
//            predictions.add(new Recognition("1", labels.get(1), probability[0]));
//            return predictions;
//        } else {
//            // Gets the map of label and probability.
//            Map<String, Float> labeledProbability =
//                    new TensorLabel(labels, outputProbabilityBuffer)
//                            .getMapWithFloatValue();
//
//            //Gets top-k results.
//            return getTopKProbability(labeledProbability);
//        }
//    }
//
//    /**
//     * Loads input image, and applies pre-processing.
//     */
//    private float[][][][] loadImage(final FrameDataHolder images) {
//
//        // Rescale to expected dimensions
//        Bitmap rgbRescaled = Bitmap.createScaledBitmap(images.rgbBitmap, imageWidth, imageHeight, true);
//        Bitmap firRescaled = Bitmap.createScaledBitmap(images.firBitmap, imageWidth, imageHeight, true);
//
//        rgbRescaled = transformer.transform(rgbRescaled);
//
//        int[] rgbArray = new int[imageWidth * imageHeight];
//        rgbRescaled.getPixels(rgbArray, 0, imageWidth, 0, 0, imageWidth, imageHeight);
//
//        int[] firArray = new int[imageWidth * imageHeight];
//        firRescaled.getPixels(firArray, 0, imageWidth, 0, 0, imageWidth, imageHeight);
//
//        // Batch size * height * width * 4 channels
//        float[][][][] mergedArray = new float[1][imageHeight][imageWidth][4];
//
//        // Extract RGB and thermal channels and combine into new array
//        for (int i = 0; i < imageWidth; i++) {
//            for (int j = 0; j < imageHeight; j++) {
//                Color rgbColor = Color.valueOf(rgbArray[imageHeight * i + j]);
//                Color firColor = Color.valueOf(firArray[imageHeight * i + j]);
//
//                mergedArray[0][j][i][0] = rgbColor.red();
//                mergedArray[0][j][i][1] = rgbColor.green();
//                mergedArray[0][j][i][2] = rgbColor.blue();
//                mergedArray[0][j][i][3] = (firColor.red() + firColor.green() + firColor.blue()) / 3;
//            }
//        }
//        return mergedArray;
//    }
//
//    /**
//     * Gets the top-k results.
//     */
//    private static List<Recognition> getTopKProbability(Map<String, Float> labelProb) {
//        // Find the best classifications.
//        PriorityQueue<Recognition> pq = new PriorityQueue<>(
//                MAX_RESULTS,
//                (lhs, rhs) -> {
//                    // Intentionally reversed to put high confidence at the head of the queue.
//                    return Float.compare(rhs.getConfidence(), lhs.getConfidence());
//                });
//
//        for (Map.Entry<String, Float> entry : labelProb.entrySet()) {
//            pq.add(new Recognition("" + entry.getKey(), entry.getKey(), entry.getValue()));
//        }
//
//        final ArrayList<Recognition> recognitions = new ArrayList<>();
//        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
//        for (int i = 0; i < recognitionsSize; ++i) {
//            recognitions.add(pq.poll());
//        }
//        return recognitions;
//    }
//
//    /**
//     * Closes the interpreter and model to release resources.
//     */
//    public void close() {
//        if (tflite != null) {
//            tflite.close();
//            tflite = null;
//        }
//        if (gpuDelegate != null) {
//            gpuDelegate.close();
//            gpuDelegate = null;
//        }
//        if (nnApiDelegate != null) {
//            nnApiDelegate.close();
//            nnApiDelegate = null;
//        }
//        tfliteModel = null;
//    }
//
//
//    /**
//     * An immutable result returned by a Classifier describing what was recognized.
//     */
//    public static class Recognition {
//        /**
//         * A unique identifier for what has been recognized. Specific to the class, not the instance of
//         * the object.
//         */
//        private final String id;
//
//        /**
//         * Display name for the recognition.
//         */
//        private final String title;
//
//        /**
//         * A sortable score for how good the recognition is relative to others. Higher should be better.
//         */
//        private final Float confidence;
//
//
//        Recognition(
//                final String id, final String title, final Float confidence) {
//            this.id = id;
//            this.title = title;
//            this.confidence = confidence;
//        }
//
//        public String getId() {
//            return id;
//        }
//
//        public String getTitle() {
//            return title;
//        }
//
//        public Float getConfidence() {
//            return confidence;
//        }
//
//        @Override
//        @NonNull
//        public String toString() {
//            String resultString = "";
//
//            if (title != null) {
//                resultString += title + " ";
//            }
//
//            if (confidence != null) {
//                resultString += String.format(Locale.UK, "(%.1f%%) ", confidence * 100.0f);
//            }
//
//            return resultString.trim();
//        }
//    }
//
//}

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;

import androidx.annotation.NonNull;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

public class BinaryClassifier {
    private Interpreter interpreter;
    private List<String> labelList;
    private int INPUT_SIZE;
    private int PIXEL_SIZE = 3;
    private float IMAGE_MEAN = 0f;
    private float IMAGE_STD = 255.0f;
    private int MAX_RESULTS = 3;
    private float THRESHOLD = 0.4f;

    BinaryClassifier(AssetManager assetManager, String modelPath, String labelPath, int inputSize) throws IOException {
        INPUT_SIZE = inputSize;
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4);
        options.setUseNNAPI(true);
        interpreter = new Interpreter(loadModelFile(assetManager, modelPath), options);
        labelList = loadLabelList(assetManager, labelPath);
    }


    class Recognition {
        String id = "";
        String title = "";
        Float confidence = 0f;

        public Recognition(String i, String t, Float c) {
            id = i;
            title = t;
            confidence = c;

        }

        @NonNull
        @Override
        public String toString() {
            String resultString = "";
            if (id != null) {
                resultString += "[" + id + "] ";
            }

            if (title != null) {
                resultString += title + " ";
            }

            if (confidence != null) {
                resultString += String.format("(%.1f%%) ", confidence * 100.0f);
            }

            return resultString.trim();
        }
    }

    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private List<String> loadLabelList(AssetManager assetManager, String labelPath) throws IOException {
        List<String> labelList = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(assetManager.open(labelPath)));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

//    List<Recognition> reconizeImage(Bitmap bitmap){
//        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);
//        ByteBuffer byteBuffer= convertBitmapToByteBuffer(scaledBitmap);
//        float [][] result = new float[1][labelList.size()];
//        interpreter.run(byteBuffer, result);
//        return getSortedResultFloat(result);
//    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap){
        ByteBuffer byteBuffer;
        byteBuffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE);

        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(intValues, 0 , bitmap.getWidth(), 0,0,bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i=0 ; i < INPUT_SIZE; i++){
            for (int j=0; j < INPUT_SIZE; j++ ) {
                final int val = intValues[pixel++];
                byteBuffer.putFloat((((val >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                byteBuffer.putFloat((((val >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                byteBuffer.putFloat((((val ) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);

            }
        }
        return byteBuffer;
    }

//    private List<Recognition> getSortedResultFloat(float[][] labelProbArray){
//        PriorityQueue<Recognition> pq = new PriorityQueue<>(
//                (int) MAX_RESULTS,
//                (Comparator<? super Recognition>)(lhs, rhs) -> {
//                    return Float.compare(rhs.confidence, lhs.confidence);
//                });
//        for (int i = 0 ; i < labelList.size(); i++){
//            float confidence = labelProbArray[0][i];
//            if (confidence > THRESHOLD){
//                pq.add(new Recognition(""+i, labelList.size() > i ? labelList.get(i): "unknown", confidence));
//            }
//        }
//        final ArrayList<Recognition>
//
//    }



}


