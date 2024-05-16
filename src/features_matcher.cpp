#include "features_matcher.h"

#include <iostream>
#include <map>

//#include "opencv2/xfeatures2d.hpp"
//#include "opencv2/xfeatures2d/nonfree.hpp"


FeatureMatcher::FeatureMatcher(cv::Mat intrinsics_matrix, cv::Mat dist_coeffs, double focal_scale)
{
  intrinsics_matrix_ = intrinsics_matrix.clone();
  dist_coeffs_ = dist_coeffs.clone();
  new_intrinsics_matrix_ = intrinsics_matrix.clone();
  new_intrinsics_matrix_.at<double>(0,0) *= focal_scale;
  new_intrinsics_matrix_.at<double>(1,1) *= focal_scale;
}

cv::Mat FeatureMatcher::readUndistortedImage(const std::string& filename )
{
  cv::Mat img = cv::imread(filename), und_img, dbg_img;
  cv::undistort	(	img, und_img, intrinsics_matrix_, dist_coeffs_, new_intrinsics_matrix_ );

  return und_img;
}

void FeatureMatcher::extractFeatures()
{
  features_.resize(images_names_.size());
  descriptors_.resize(images_names_.size());
  feats_colors_.resize(images_names_.size());

  //Used to select the descriptor: ORB,BRISK,AKAZE,SURF
  int feature_detector_type = 2;
  
  int hessian = 10;
  int minKey = 12000;
  cv::Ptr<cv::ORB> orb_detector = cv::ORB::create(minKey); //0
  cv::Ptr<cv::BRISK> brisk_detector = cv::BRISK::create(); //1
  cv::Ptr<cv::AKAZE> akaze_detector = cv::AKAZE::create(); //2
  akaze_detector->setThreshold(0.0001);
  akaze_detector->setNOctaves(8);
  akaze_detector->setNOctaveLayers(6);
  //cv::Ptr<cv::xfeatures2d::SURF> surf_detector= cv::xfeatures2d::SURF::create(hessian); //3

  for( int i = 0; i < images_names_.size(); i++  )
  {
    std::cout<<"Computing descriptors for image "<<i<<std::endl;
    cv::Mat img = readUndistortedImage(images_names_[i]);

    //////////////////////////// Code to be completed (1/7) /////////////////////////////////
    // Extract salient points + descriptors from i-th image, and store them into
    // features_[i] and descriptors_[i] vector, respectively
    // Extract also the color (i.e., the cv::Vec3b information) of each feature, and store
    // it into feats_colors_[i] vector
    /////////////////////////////////////////////////////////////////////////////////////////
    
    //Convert to grey image, better results (and fast)
    cv::Mat grey;
    cv::cvtColor(img,grey,cv::COLOR_BGR2GRAY);

    switch(feature_detector_type){

      case 0:
        orb_detector->detectAndCompute(grey, cv::noArray(), features_[i], descriptors_[i]);
        break;
      
      case 1:
        brisk_detector->detectAndCompute(grey, cv::noArray(), features_[i], descriptors_[i]);
        break;
      
      case 2:
        akaze_detector->detectAndCompute(grey, cv::noArray(), features_[i], descriptors_[i]);
        break;

     // case 3:
       // surf_detector->detectAndCompute(grey, cv::noArray(), features_[i], descriptors_[i]);
        //break;
    
    }
    
    for(int j = 0; j < features_[i].size(); j++){

      cv::Vec3b color = img.at<cv::Vec3b>(features_[i][j].pt);
      feats_colors_[i].push_back(color);
    }

    /////////////////////////////////////////////////////////////////////////////////////////
  }
}

      
void FeatureMatcher::exhaustiveMatching()
{
 
  std::vector<cv::DMatch> matches, inlier_matches; //for each couple of images

  for( int i = 0; i < images_names_.size() - 1; i++ )
  {
    for( int j = i + 1; j < images_names_.size(); j++ )
    {
      
      std::cout<<"Matching image "<<i<<" with image "<<j<<std::endl;

      //////////////////////////// Code to be completed (2/7) /////////////////////////////////
      // Match descriptors between image i and image j, and perform geometric validation,
      // possibly discarding the outliers (remember that features have been extracted
      // from undistorted images that now has new_intrinsics_matrix_ as K matrix and
      // no distortions)
      // As geometric models, use both the Essential matrix and the Homograph matrix,
      // both by setting new_intrinsics_matrix_ as K matrix.
      // As threshold in the functions to estimate both models, you may use 1.0 or similar.
      // Store inlier matches into the inlier_matches vector
      // Do not set matches between two images if the amount of inliers matches
      // (i.e., geomatrically verified matches) is small (say <= 5 matches)
      // In case of success, set the matches with the function:
      // setMatches( i, j, inlier_matches);
      /////////////////////////////////////////////////////////////////////////////////////////


      int min_matches = 5;

      //Create two different approach for the matcher
      //The first for AKAZE and the second for SURF, ORB
      //To use one of them comment the other one
      
      //AKAZE
      
      std::vector<std::vector<cv::DMatch>> knn_matches;
      cv::BFMatcher matcher(cv::NORM_HAMMING);
      matcher.knnMatch(descriptors_[i], descriptors_[j], knn_matches, 2);

      const float ratio_threshold = 0.8f;
      std::vector<cv::Point2f> points_i, points_j;
      for (const auto &m: knn_matches) 
      {
          if (m[0].distance < ratio_threshold * m[1].distance) 
          { 
            matches.push_back(m[0]);
            points_i.push_back(features_[i][m[0].queryIdx].pt);
            points_j.push_back(features_[j][m[0].trainIdx].pt);
          }
      }
      
      
      

      //SURF & ORB
      /*
      cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_L2,false);
      matcher->match(descriptors_[i], descriptors_[j], matches);

      std::vector<cv::Point2f> points_i, points_j;
      for(int k = 0; k < matches.size(); k++ )
      {
        points_i.push_back(features_[i][matches[k].queryIdx].pt);
        points_j.push_back(features_[j][matches[k].trainIdx].pt);
      }
      */
      
      
      //Compute the essential matrix and the homography matrix
      cv::Mat mask_essential, mask_homography;
      cv::findEssentialMat(points_i, points_j, new_intrinsics_matrix_, cv::RANSAC, 0.999, 1.0,mask_essential);
      cv::findHomography(points_i, points_j,mask_homography, cv::RANSAC, 1.0);


      std::vector<cv::DMatch> inlier_matches_H,inlier_matches_E;

      //We perform the geometric validation and select the model that gives us 
      //the highest number of inliers
      for(int k=0;k<matches.size();k++)
      {
        if(mask_homography.at<uchar>(k,0)==1)
        {
          inlier_matches_H.push_back(matches[k]);
        }
        if(mask_essential.at<uchar>(k,0)==1)
        {
          inlier_matches_E.push_back(matches[k]);
        }
      }

      int H = inlier_matches_H.size();
      int E = inlier_matches_E.size();

      if(H > E)
      {
        for (int t = 0; t < inlier_matches_H.size(); t++)
        {
          inlier_matches.push_back(inlier_matches_H[t]);
        }
      }
      else 
      {
        for (int t = 0; t < inlier_matches_E.size(); t++)
        {
          inlier_matches.push_back(inlier_matches_E[t]);
        }
      }

      //Set the matches after checking the minimum size = 5
      if (inlier_matches.size() > min_matches)
      {
        setMatches(i, j, inlier_matches);
      }

      matches.clear();
      inlier_matches.clear();

      /////////////////////////////////////////////////////////////////////////////////////////
    }
  }
}

void FeatureMatcher::writeToFile ( const std::string& filename, bool normalize_points ) const
{
  FILE* fptr = fopen(filename.c_str(), "w");

  if (fptr == NULL) {
    std::cerr << "Error: unable to open file " << filename;
    return;
  };

  fprintf(fptr, "%d %d %d\n", num_poses_, num_points_, num_observations_);

  double *tmp_observations;
  cv::Mat dst_pts;
  if(normalize_points)
  {
    cv::Mat src_obs( num_observations_,1, cv::traits::Type<cv::Vec2d>::value,
                     const_cast<double *>(observations_.data()));
    cv::undistortPoints(src_obs, dst_pts, new_intrinsics_matrix_, cv::Mat());
    tmp_observations = reinterpret_cast<double *>(dst_pts.data);
  }
  else
  {
    tmp_observations = const_cast<double *>(observations_.data());
  }

  for (int i = 0; i < num_observations_; ++i)
  {
    fprintf(fptr, "%d %d", pose_index_[i], point_index_[i]);
    for (int j = 0; j < 2; ++j) {
      fprintf(fptr, " %g", tmp_observations[2 * i + j]);
    }
    fprintf(fptr, "\n");
  }

  if( colors_.size() == 3*num_points_ )
  {
    for (int i = 0; i < num_points_; ++i)
      fprintf(fptr, "%d %d %d\n", colors_[i*3], colors_[i*3 + 1], colors_[i*3 + 2]);
  }

  fclose(fptr);
}

void FeatureMatcher::testMatches( double scale )
{
  // For each pose, prepare a map that reports the pairs [point index, observation index]
  std::vector< std::map<int,int> > cam_observation( num_poses_ );
  for( int i_obs = 0; i_obs < num_observations_; i_obs++ )
  {
    int i_cam = pose_index_[i_obs], i_pt = point_index_[i_obs];
    cam_observation[i_cam][i_pt] = i_obs;
  }

  for( int r = 0; r < num_poses_; r++ )
  {
    for (int c = r + 1; c < num_poses_; c++)
    {
      int num_mathces = 0;
      std::vector<cv::DMatch> matches;
      std::vector<cv::KeyPoint> features0, features1;
      for (auto const &co_iter: cam_observation[r])
      {
        if (cam_observation[c].find(co_iter.first) != cam_observation[c].end())
        {
          features0.emplace_back(observations_[2*co_iter.second],observations_[2*co_iter.second + 1], 0.0);
          features1.emplace_back(observations_[2*cam_observation[c][co_iter.first]],observations_[2*cam_observation[c][co_iter.first] + 1], 0.0);
          matches.emplace_back(num_mathces,num_mathces, 0);
          num_mathces++;
        }
      }
      cv::Mat img0 = readUndistortedImage(images_names_[r]),
          img1 = readUndistortedImage(images_names_[c]),
          dbg_img;

      cv::drawMatches(img0, features0, img1, features1, matches, dbg_img);
      cv::resize(dbg_img, dbg_img, cv::Size(), scale, scale);
      cv::imshow("", dbg_img);
      if (cv::waitKey() == 27)
        return;
    }
  }
}

void FeatureMatcher::setMatches( int pos0_id, int pos1_id, const std::vector<cv::DMatch> &matches )
{

  const auto &features0 = features_[pos0_id];
  const auto &features1 = features_[pos1_id];

  auto pos_iter0 = pose_id_map_.find(pos0_id),
      pos_iter1 = pose_id_map_.find(pos1_id);

  // Already included position?
  if( pos_iter0 == pose_id_map_.end() )
  {
    pose_id_map_[pos0_id] = num_poses_;
    pos0_id = num_poses_++;
  }
  else
    pos0_id = pose_id_map_[pos0_id];

  // Already included position?
  if( pos_iter1 == pose_id_map_.end() )
  {
    pose_id_map_[pos1_id] = num_poses_;
    pos1_id = num_poses_++;
  }
  else
    pos1_id = pose_id_map_[pos1_id];

  for( auto &match:matches)
  {

    // Already included observations?
    uint64_t obs_id0 = poseFeatPairID(pos0_id, match.queryIdx ),
        obs_id1 = poseFeatPairID(pos1_id, match.trainIdx );
    auto pt_iter0 = point_id_map_.find(obs_id0),
        pt_iter1 = point_id_map_.find(obs_id1);
    // New point
    if( pt_iter0 == point_id_map_.end() && pt_iter1 == point_id_map_.end() )
    {
      int pt_idx = num_points_++;
      point_id_map_[obs_id0] = point_id_map_[obs_id1] = pt_idx;

      point_index_.push_back(pt_idx);
      point_index_.push_back(pt_idx);
      pose_index_.push_back(pos0_id);
      pose_index_.push_back(pos1_id);
      observations_.push_back(features0[match.queryIdx].pt.x);
      observations_.push_back(features0[match.queryIdx].pt.y);
      observations_.push_back(features1[match.trainIdx].pt.x);
      observations_.push_back(features1[match.trainIdx].pt.y);

      // Average color between two corresponding features (suboptimal since we shouls also consider
      // the other observations of the same point in the other images)
      cv::Vec3f color = (cv::Vec3f(feats_colors_[pos0_id][match.queryIdx]) +
                        cv::Vec3f(feats_colors_[pos1_id][match.trainIdx]))/2;

      colors_.push_back(cvRound(color[2]));
      colors_.push_back(cvRound(color[1]));
      colors_.push_back(cvRound(color[0]));

      num_observations_++;
      num_observations_++;
    }
      // New observation
    else if( pt_iter0 == point_id_map_.end() )
    {
      int pt_idx = point_id_map_[obs_id1];
      point_id_map_[obs_id0] = pt_idx;

      point_index_.push_back(pt_idx);
      pose_index_.push_back(pos0_id);
      observations_.push_back(features0[match.queryIdx].pt.x);
      observations_.push_back(features0[match.queryIdx].pt.y);
      num_observations_++;
    }
    else if( pt_iter1 == point_id_map_.end() )
    {
      int pt_idx = point_id_map_[obs_id0];
      point_id_map_[obs_id1] = pt_idx;

      point_index_.push_back(pt_idx);
      pose_index_.push_back(pos1_id);
      observations_.push_back(features1[match.trainIdx].pt.x);
      observations_.push_back(features1[match.trainIdx].pt.y);
      num_observations_++;
    }
//    else if( pt_iter0->second != pt_iter1->second )
//    {
//      std::cerr<<"Shared observations does not share 3D point!"<<std::endl;
//    }
  }
}
void FeatureMatcher::reset()
{
  point_index_.clear();
  pose_index_.clear();
  observations_.clear();
  colors_.clear();

  num_poses_ = num_points_ = num_observations_ = 0;
}
