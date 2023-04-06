add_executable(ex3 ex3.cpp)
target_link_libraries(ex3 hyperwavelet ~/local/lib/libmetis.so libGKlib.a)
install(TARGETS ex3 DESTINATION /home/dmitriy/HyperWavelet/examples/ex3/)