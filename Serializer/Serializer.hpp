/*
 * Serializer.h
 *
 *  Created on: Aug 16, 2019
 *      Author: sakex
 */

#ifndef SERIALIZER_SERIALIZER_H_
#define SERIALIZER_SERIALIZER_H_

#include "Serializable.h"
#include <fstream>
#include <string>
#include <iostream>

#include <type_traits>

namespace Serializer {

    template<bool B>
    struct is_serializable {
    };

    template<>
    struct is_serializable<false> {
        static void to_file(std::string const &output, std::string const &file_name) {
            std::ofstream file(file_name);
            file << output;
            file.close();
        }
    };

    template<>
    struct is_serializable<true> {
        template<typename T>
        static void to_file(T *serializable, std::string const &file_name) {
            std::string output = serializable->to_string();
            is_serializable<false>::to_file(output, file_name);
        }
    };

    template<typename T>
    void to_file(T *to_print, std::string const &file_name) {
        is_serializable<std::is_base_of<Serializable, T>::value>::to_file(to_print, file_name);
    }

    template<typename T>
    void to_file(T to_print, std::string const &file_name) {
        is_serializable<std::is_base_of<Serializable, T>::value>::to_file(to_print, file_name);
    }


} /* namespace Serializer */

#endif /* SERIALIZER_SERIALIZER_H_ */
