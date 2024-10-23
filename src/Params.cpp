#include "Params.h"
using namespace std;

std::map<std::string, std::vector<std::string>> classes_map;
void Params_init()
{
    classes_map["WhHandGesture_class"] = {"Left_Slid","Right_Slid","Thump_Down","Thump_Up","Stop","Zoom_In","Zoom_Out"};
    classes_map["IsHandGesture_class"] = {"No_gesture", "Is_Gesture"};
}