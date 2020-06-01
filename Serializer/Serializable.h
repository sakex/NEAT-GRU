/*
 * Serializable.h
 *
 *  Created on: Aug 16, 2019
 *      Author: sakex
 */

#ifndef SERIALIZER_SERIALIZABLE_H_
#define SERIALIZER_SERIALIZABLE_H_

#include <string>

/// Serializer namespace
namespace Serializer {

    /// Serializable abstract class, implement to use to_file()
    class Serializable {
    public:
        Serializable() = default;

        virtual ~Serializable() = default;

        /**
         * Convert Serializable to string
         * @return string representation of the Serializable
         */
        std::string to_string() const;

    private:
        /**
         * Pure virtual method to implement to convert Serializable to string
         * @return string representation of the Serializable
         */
        virtual std::string parse_to_string() const = 0;
    };

} /* namespace Serializer */

#endif /* SERIALIZER_SERIALIZABLE_H_ */
