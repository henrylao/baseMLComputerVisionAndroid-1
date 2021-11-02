//package com.grewe.ilab.opencv401;    //THIS will change and be YOUR package name
package edu.ilab.cs663.edge;

import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import edu.ilab.cs663.R;

/**
 * Activity class demonstrating edge detection using Sobel's filter
 */
public class ImageProcessActivity extends AppCompatActivity implements CvCameraViewListener2 {

    // will point to our View widget for our image
    private CameraBridgeViewBase mOpenCvCameraView;
    private static final String TAG = "OCVSample::Activity";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);

        //Load in the OpenCV dependency module code from the jni files you linked in this project
        // inside the OpenCV module
        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(ImageProcessActivity.this, "Unable to load OpenCV", Toast.LENGTH_LONG).show();
        }
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_image_process);

        // grab a "handle" to the OpenCV class responsible for viewing Image
        // look at the XML the id of our CameraBridgeViewBase is HelloOpenCVView
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.HelloOpenCVCameraView);
        //set it visible, register the listener and enbale the view so connected to camera
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this); // the activity will listen to events on Camera -call onCameraFrame
        mOpenCvCameraView.enableView();
    }

    // disable JavaCameraView if app going on pause
    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    //enable the JavaCameraView if app resuming
    @Override
    public void onResume() {
        super.onResume();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.enableView();
    }


    //Disable view of JavaCameraView if app is being destoryed
    @Override
    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }


    //method invoked when camera view is started
    public void onCameraViewStarted(int width, int height) {

    }


    //method invoked when camera view is stoped
    public void onCameraViewStopped() {

    }

    /**
     * THIS IS THE main method that is called each time you get a new Frame/Image
     * Implement to be a CVCameraViewListener2
     * @param inputFrame
     * @return
     */
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        //store in the imageMat (instance of OpenCV's Mat class, a 2D matrix) the RGB(A=alpha) image
        Mat imageMat = inputFrame.rgba();

       /* note in an Java API based capture Activity you will have a method that is invoked when a
          new frame is digitized and you will need to convert the image frame datastructure  object
          from Java api to an OpenCV Mat object instance so it can be processed with OpenCV calls
          Typically this means converting a Bitmap to an OpenCV Mat.
             Mat  openCVMatImage =  new Mat (bmp.getWidth(), bmp.getHeight(), CvType.CV_8UC1);
             Utils.bitmapToMat(bmp, openCVMatImage);
          Then you can process it like  the following line that converts an rgb image to greyscale
           in this call I am storing the results in the same Mat object
              mgproc.cvtColor(openCVMatImage,openCVMatImage, Imgproc.COLOR_RGB2GRAY);
       */

        Mat gray = inputFrame.gray();
        Mat mIntermediateMat = new Mat();
        Imgproc.Sobel(gray, mIntermediateMat, CvType.CV_8U, 1, 1);
        Core.convertScaleAbs(mIntermediateMat, mIntermediateMat, 10, 0);
        Imgproc.cvtColor(mIntermediateMat, imageMat, Imgproc.COLOR_GRAY2BGRA, 4);

        //Return the Mat you want to be displayed in the JavaCameraView widget which invoked this method
        return imageMat;
    }
    
    public void resetFilterMode(){

    }

}
