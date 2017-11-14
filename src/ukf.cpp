#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = false;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30; 

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  ///* initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;
    
  ///* State dimension
  n_x_ = 5;
    
  ///* Augmented state dimension
  n_aug_ = 7;

  // initial state vector
  x_ = VectorXd(n_x_);
    
  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);
  
  ///* Weights of sigma points
  weights_ = VectorXd(2 * n_aug_ + 1);

  ///* Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  ///* predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);

  ///* time when the state is true, in us
  time_us_ = 0;

  ///* NIS Lidar
  NIS_lidar_ = 0;

  ///* NIS Radar
  NIS_radar_ = 0;

  // Measurement noise covariance matrix initialization
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << std_radr_*std_radr_,  0,    0,
              0, std_radphi_*std_radphi_, 0,
              0, 0,   std_radrd_*std_radrd_;

  R_lidar_ = MatrixXd(2, 2);
  R_lidar_ << std_laspx_*std_laspx_,0,
              0,std_laspy_*std_laspy_;
  }

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(const MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  /*************
  Ininitialization 
  **************/
  if (!is_initialized_){
    /**
      * Initialize the state x_ with the first measurement and P_
      * Create the covariance matrix.
      * convert radar from polar to cartesian coordinates.
    */
    // first measurement
   
    // Initialized state_x
    x_.fill(0.0);
    // INitialized covariance matrix P
    P_ << 1, 0, 0, 0, 0,
          0, 1, 0, 0, 0,
          0, 0, 1, 0, 0,
          0, 0, 0, 1, 0,
          0, 0, 0, 0, 1;

    if(meas_package.sensor_type_ == MeasurementPackage::LASER){
      x_(0) = meas_package.raw_measurements_(0);
      x_(1) = meas_package.raw_measurements_(1);  
    }
    if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
      float ro = meas_package.raw_measurements_(0);
      float phi = meas_package.raw_measurements_(1);
      float ro_dot = meas_package.raw_measurements_(2);
      // Coordinates convertion from polar to cartesian
      float px = ro * cos(phi); 
      float py = ro * sin(phi);
      float vx = ro_dot * cos(phi);
      float vy = ro_dot * sin(phi);
      float v  = sqrt(vx * vx + vy * vy);
    x_ << px, py, v, 0, 0;
    }

    // Initialize weights
    weights_(0) = lambda_ / (lambda_ + n_aug_);
    for (int i = 1; i < weights_.size(); i++) {
        weights_(i) = 0.5 / (n_aug_ + lambda_);
    }

    // set up initial timestamp
    time_us_ = meas_package.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    cout << "UKF initialized" << endl;
    return;    
  }
  // prediction
  double delta_t = (meas_package.timestamp_ - time_us_)/1000000.0; // s
  time_us_ = meas_package.timestamp_;
  Prediction(delta_t);
  cout << "prediction work!" << endl;
  

  //update
  if(meas_package.sensor_type_ == MeasurementPackage::LASER){
    cout<<"detect lidar"<<endl;
    UpdateLidar(meas_package);
    cout<<"update lidar"<<endl;
  }
  if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
    cout<<"detect radar"<<endl;
    UpdateRadar(meas_package);
    cout<<"update radar"<<endl;
  }
}

/**
#include "ukf.h"
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /************************
  1. generate sigma points
  *************************/
  //noise augmentatation
  //new lambda value
  lambda_ = 3 - n_aug_;
  
  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);
  
  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  
  //create augmented mean state, noise mean state is zero
  x_aug.head(n_x_) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;
  
  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_ * std_a_;
  P_aug(6,6) = std_yawdd_ * std_yawdd_;
  
  //create square root matrix
  MatrixXd A = P_aug.llt().matrixL();
  
  //create augmented sigma points 7*15
  Xsig_aug.col(0) = x_aug;
  for(int i = 0; i<n_aug_; i++){
      Xsig_aug.col(i+1) = x_aug + sqrt(lambda_ + n_aug_) * A.col(i);
      Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * A.col(i);
  }

  /************************
  2. predict sigma points
  *************************/
  for (int i = 0; i< 2*n_aug_+1; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v; 
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column 5*15
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
  /************
   3. predict mean and covariance
   * **********/
  //predict state mean
  x_.fill(0.0); //reset prdicted sigma points mean
  for (int i = 0; i<2*n_aug_+1; i++){
    x_ +=  weights_(i)*Xsig_pred_.col(i);
  }
  
  //predict state covariance matrix  
  P_.fill(0.0); //reset predicted sigma points coviance
  for (int n = 0; n<2*n_aug_+1; n++){
    VectorXd x_diff = VectorXd(n_x_);
    x_diff = Xsig_pred_.col(n)-x_;
    
    // normalized the angle
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;     
    
    P_ += weights_(n)*x_diff*x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  //1. predict measurement
  int n_z = 2;

  MatrixXd Zsig = MatrixXd(n_z, 2*n_aug_+1);
  for(int i = 0; i<2 * n_aug_+1; i++){
    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    // measurement model
    //Zsig.col(i) << p_x, p_y;
    Zsig(0, i) = p_x;
    Zsig(1, i) = p_y;
  }
  updateUKF(Zsig, n_z, meas_package);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {

/************************
 * predict measurement
 * *********************/
  int n_z = 3;
  MatrixXd Zsig = MatrixXd(n_z, 2*n_aug_+1); 
  for (int i = 0; i < 2*n_aug_+1; i++) {  //2n+1 simga points
    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);
    
    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig(1,i) = atan2(p_y,p_x);                                 //phi
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
    }
    updateUKF(Zsig, n_z, meas_package);
    cout<<"Radar update measurement ok"<<endl;
}


void UKF::updateUKF(MatrixXd Zsig, int n_z, MeasurementPackage meas_package){
     
  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }
  cout<<"z: "<<z_pred<<endl;
  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  //VectorXd z_diff = VectorXd(n_z);
  S.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // angle normalization for Radar measurements
    if (meas_package.sensor_type_ = MeasurementPackage::RADAR){
      while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
      while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
    }
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
  cout<<"S is "<<S<<endl;
  // Add measurement noise covariance matrix
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR){ // Radar
    S = S + R_radar_;
    cout<<"s with radar noisy is "<<S<<endl;
  }
  if (meas_package.sensor_type_ == MeasurementPackage::LASER){ // Lidar
    S = S + R_lidar_;
    cout<<"s with lidar noisy is "<<S<<endl;
  }
  /************************
   2. update
  * *********************/
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1; i++) {  //2n+1 simga points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // angle normalization for Radar measurements
    if (meas_package.sensor_type_ = MeasurementPackage::RADAR){
      while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
      while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
    }
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
  cout<<"TC: "<<Tc<<endl;
  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  cout<<"K: "<<K<<endl;
  //extract measurements
  VectorXd z = meas_package.raw_measurements_;
  cout<<"z: "<<z<<endl;
  //residual
  VectorXd z_diff = z - z_pred;
  cout<<"z - z_pred = : "<<z_diff<<endl;

  // angle normalization for Radar measurements
  if (meas_package.sensor_type_ = MeasurementPackage::RADAR){
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
  }

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
/*
  //calculate NIS
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR){
    NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
    }
  if (meas_package.sensor_type_ == MeasurementPackage::LASER){
    NIS_lidar_ = z_diff.transpose() * S.inverse() * z_diff;
  }
  */
}
