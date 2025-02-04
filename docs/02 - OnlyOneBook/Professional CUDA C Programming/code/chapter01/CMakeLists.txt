add_executable(helloWorld helloWorld.cu)

target_compile_options(helloWorld PRIVATE -g -G -O3)

install(TARGETS helloWorld DESTINATION chapter01)