all:
	cd rcnn/cython/; python setup.py build_ext --inplace; rm -rf build; cd ../../

	cd rcnn/dataset/; python setup.py build_ext --inplace; rm -rf build; cd ../../
clean:
	cd rcnn/cython/; rm *.so *.c *.cpp; cd ../../
	
	cd rcnn/dataset/external/; rm *.so; cd ../../../
