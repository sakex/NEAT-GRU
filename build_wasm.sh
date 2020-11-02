cd build
emcmake cmake .. -DCMAKE_BUILD_TYPE=Release
emcmake make
emcc libNEAT_GRU.a -s EXTRA_EXPORTED_RUNTIME_METHODS=['ccall'] -s ENVIRONMENT='web' -s EXPORT_ES6=1 -s MODULARIZE=1 -s USE_ES6_IMPORT_META=0 \
-s EXPORTED_FUNCTIONS="['_compute_network', '_reset_network_state', '_network_from_string', '_network_from_topology', '_topology_to_string', '_fit', '_topology_delta_compatibility']" -o neat_gru.js
