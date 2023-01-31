#include "handler_function.h"

std::vector<std::string> read_data_from_file(const char *File_Name){
		std::ifstream input(File_Name);
		std::string line;
		std::vector<std::string> str_arr;
	    while (std::getline(input, line))
	    {
	    	str_arr.push_back(line);
	    }
		return str_arr;
}
