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

/// Serializer namespace
namespace Serializer {

    /**
     * Static method to check if a class implements Serializable
     * @tparam Serializable
     */
    template<bool Serializable>
    struct is_serializable {
    };

    /// If is not serializable, try to use the ofstream overload
    template<>
    struct is_serializable<false> {
        static void to_file(std::string const &output, std::string const &file_name) {
            std::ofstream file(file_name);
            file << output;
            file.close();
        }
    };

    /// If serializable, run to_file function with to_string() implementation
    template<>
    struct is_serializable<true> {
        template<typename T>
        static void to_file(T *serializable, std::string const &file_name) {
            std::string output = serializable->to_string();
            is_serializable<false>::to_file(output, file_name);
        }
    };

    /**
     * Exports to file from pointer
     *
     * @tparam T Type of the object to be serialized
     * @param to_print Pointer to the object to be serialized
     * @param file_name File to which the string should be written
     */
    template<typename T>
    void to_file(T *to_print, std::string const &file_name) {
        is_serializable<std::is_base_of<Serializable, T>::value>::to_file(to_print, file_name);
    }

    /**
     * Exports to file from value
     *
     * @tparam T Type of the object to be serialized
     * @param to_print Object to be serialized
     * @param file_name File to which the string should be written
     */
    template<typename T>
    void to_file(T to_print, std::string const &file_name) {
        is_serializable<std::is_base_of<Serializable, T>::value>::to_file(to_print, file_name);
    }


} /* namespace Serializer */

#endif /* SERIALIZER_SERIALIZER_H_ */
