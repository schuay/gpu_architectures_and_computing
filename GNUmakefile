bake:
	mkdir -p build
	cd build; cmake -DCMAKE_BUILD_TYPE="Debug" .. && make -j6 && ctest --output-on-failure

clean:
	rm -rf build
