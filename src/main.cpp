#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

#include <pcl/recognition/linemod.h>
#include <pcl/recognition/color_gradient_modality.h>
#include <pcl/recognition/surface_normal_modality.h>

#include <boost/range/irange.hpp>
#include <boost/range/counting_range.hpp>
#include <boost/smart_ptr/make_shared.hpp>


// Typedefs
using PointType = pcl::PointXYZRGBA;
using Cloud = pcl::PointCloud <PointType>;

//const auto& range = boost::irange;

namespace fs = boost::filesystem;


// User defined literals
// @formatter:off
constexpr size_t operator "" _sz (unsigned long long size) { return size_t{size}; }
constexpr double operator "" _deg (long double deg) { return deg * M_PI / 180.0; }
constexpr double operator "" _deg (unsigned long long deg) { return deg * M_PI / 180.0; }
constexpr double operator "" _cm (long double cm) { return cm / 100.0; }
constexpr double operator "" _cm (unsigned long long cm) { return cm / 100.0; }
constexpr double operator "" _mm (long double mm) { return mm / 1000.0; }
constexpr double operator "" _mm (unsigned long long mm) { return mm / 1000.0; }
// @formatter:on


// Constants
constexpr auto MIN_VALID_ARGS = 3U;
constexpr auto MAX_VALID_ARGS = 9U;
//constexpr auto NUM_PCD_FILES_EXPECTED = 2U;
//constexpr auto NUM_PCD_DIRS_EXPECTED = 2U;
constexpr auto INPUT_DIR_ARG_POS = 1U;
//constexpr auto OUTPUT_DIR_ARG_POS = 2U;
//constexpr auto DEFAULT_MIN_CLUSTER_SIZE = 100U;
//constexpr auto DEFAULT_MAX_CLUSTER_SIZE = 25000U;
//constexpr auto DEFAULT_TOLERANCE = 2_cm;
//constexpr auto DEFAULT_MAX_NUM_CLUSTERS = 20U;

template <typename T>
constexpr auto izrange (T upper)
-> decltype (auto) {
  return boost::irange (static_cast <T> (0), upper);
}


auto printHelp (int argc, char ** argv)
-> void {
  using pcl::console::print_error;
  using pcl::console::print_info;

  // TODO: Update this help
  print_error ("Syntax is: %s (<path-to-pcd-files> <path-to-output_lmt_file>) <options> | "
                   "-h | --help\n", argv[0]);
  print_info ("%s -h | --help : shows this help\n", argv[0]);
  //  print_info ("-min X : use a minimum of X points per cluster (default: 100)\n");
  //  print_info ("-max X : use a maximum of X points per cluster (default: 25000)\n");
  //  print_info ("-tol X : the spatial distance (in meters) between clusters (default: 0.002.\n");
}


auto expandTilde (std::string path_string) -> fs::path {
  if (path_string.at (0) == '~')
    path_string.replace (0, 1, getenv ("HOME"));
  return fs::path{path_string};
}


auto getPcdFilesInPath (fs::path const & pcd_dir)
-> std::vector <fs::path> {
  auto result_set = std::vector <fs::path>{};
  for (auto const & entry : boost::make_iterator_range (fs::directory_iterator{pcd_dir})) {
    if (fs::is_regular_file (entry.status ())) {
      if (entry.path ().extension () == ".pcd") {
        result_set.emplace_back (entry);
      }
    }
  }
  return result_set;
}


auto checkValidOutputFile (fs::path const & filepath)
-> bool {
  // Check that the file is valid
  auto parent_path = filepath.parent_path ();
  return fs::exists (parent_path) && fs::is_directory (parent_path);
}


constexpr auto checkValidPoint (PointType const & point)
-> bool {
  return (pcl_isfinite (point.x) && pcl_isfinite (point.y) && pcl_isfinite (point.z));
}


/**
 * This assumes that the (organised) cloud only contains foreground points.
 *
 * returns a vector of bools representing the valid points in the cloud
 * or an empty vector if the cloud is not organised
 */
auto getForegroundMask (Cloud::ConstPtr const & cloud)
-> std::vector <bool> {
  if (!cloud->isOrganized ())
    return std::vector <bool>{};

  auto mask = std::vector <bool> (cloud->size (), false);
  auto index = 0_sz;
  for (auto const & point : cloud->points) {
    if (checkValidPoint (point))
      mask.at (index) = true;
    ++index;
  }
  return mask;
}


auto trainAndAddTemplate (Cloud::ConstPtr const & input,
                          std::vector <bool> const & foreground_mask,
                          pcl::LINEMOD & linemod)
-> void {
  auto color_grad_mod = pcl::ColorGradientModality <PointType> {};
  color_grad_mod.setInputCloud (input);
  color_grad_mod.processInputData ();

  auto surface_norm_mod = pcl::SurfaceNormalModality <pcl::PointXYZRGBA> {};
  surface_norm_mod.setInputCloud (input);
  surface_norm_mod.processInputData ();

  auto modalities = std::vector <pcl::QuantizableModality *> (2);
  modalities[0] = &color_grad_mod;
  modalities[1] = &surface_norm_mod;

  auto min_x = input->width;
  auto min_y = input->height;
  auto max_x = decltype (min_x) {0};
  auto max_y = decltype (min_y) {0};

  auto mask_map = pcl::MaskMap {input->width, input->height};

  for (auto const j : izrange (input->height)) {
    for (auto const i : izrange (input->width)) {
      auto const & mask_el = foreground_mask[j * input->width + i];
      mask_map (i, j) = static_cast <unsigned char> (mask_el);

      if (mask_el) {
        min_x = std::min (min_x, i);
        max_x = std::max (max_x, i);
        min_y = std::min (min_y, j);
        max_y = std::max (max_y, j);
      }
    }
  }

  auto masks = std::vector <pcl::MaskMap *> (2);
  masks[0] = &mask_map;
  masks[1] = &mask_map;

  auto region = pcl::RegionXY {};
  region.x = static_cast<int> (min_x);
  region.y = static_cast<int> (min_y);
  region.width = static_cast<int> (max_x - min_x + 1);
  region.height = static_cast<int> (max_y - min_y + 1);

  auto ss = std::stringstream {};
  ss << "Object region: (" << region.x << ", " << region.y << ", " <<
      region.x + region.width << ", " << region.y + region.height << ") "<< std::endl;
  std::cout << ss.str ();

  linemod.createAndAddTemplate (modalities, masks, region);
}


auto getTemplates (std::vector <fs::path> & pcd_files)
-> pcl::LINEMOD {
  auto linemod = pcl::LINEMOD {};

  for (auto const & pcd_file : pcd_files) {
    auto const input_cloud = boost::make_shared <Cloud> ();
    if (pcl::io::loadPCDFile <pcl::PointXYZRGBA> (pcd_file.c_str (), *input_cloud) == -1) {
      pcl::console::print_error ("Failed to load: %s\n", pcd_file);
      continue;
    }

    auto const & foreground_mask = getForegroundMask (input_cloud);

    if (foreground_mask.size () == 0) {
      pcl::console::print_error ("No foreground points found: %s\n", pcd_file);
      continue;
    }

    trainAndAddTemplate (input_cloud, foreground_mask, linemod);
  }

  return linemod;
}


auto saveTemplates (pcl::LINEMOD const & linemod, fs::path const & output_lmt_file) {
  linemod.saveTemplates (output_lmt_file.c_str ());
}


auto main (int argc, char * argv[])
-> int {
  pcl::console::print_highlight ("Tool to extract the largest cluster found in a point cloud.\n");

  auto help_flag_1 = pcl::console::find_switch (argc, argv, "-h");
  auto help_flag_2 = pcl::console::find_switch (argc, argv, "--help");

  if (help_flag_1 || help_flag_2) {
    printHelp (argc, argv);
    return -1;
  }

  if (argc > MAX_VALID_ARGS || argc < MIN_VALID_ARGS) {
    pcl::console::print_error ("Invalid number of arguments.\n");
    printHelp (argc, argv);
    return -1;
  }

  // Check if we are working with individual files
  auto const lmt_arg_indices = pcl::console::parse_file_extension_argument (argc, argv, ".lmt");

  if (lmt_arg_indices.size () != 1) {
    pcl::console::print_error ("Invalid number of arguments.\n");
    printHelp (argc, argv);
    return -1;
  }

  auto lmt_file = fs::path {argv[lmt_arg_indices.at (0)]};

  if (!checkValidOutputFile (lmt_file)) {
    pcl::console::print_error ("A valid output file was not specified.\n");
    printHelp (argc, argv);
    return -1;
  }

  auto const input_dir = expandTilde (std::string {argv[INPUT_DIR_ARG_POS]});
  if (!fs::exists (input_dir) || !fs::is_directory (input_dir)) {
    pcl::console::print_error ("A valid input directory was not specified.\n");
    printHelp (argc, argv);
    return -1;
  }

  auto input_files = getPcdFilesInPath (input_dir);

  auto templates = getTemplates (input_files);

  saveTemplates (templates, lmt_file);

  templates.loadTemplates (lmt_file.c_str ());

  std::cout << templates.getNumOfTemplates ();

  return (0);
}