## Build Script:

Run the following script from the root directory

`cmake -B build && cmake --build build && mv -f build/Debug/linear.cp313-win_amd64.pyd linear.cp313-win_amd64.pyd`

## Notes

You may also need to download the mpreal.h header file from [here](https://github.com/advanpix/mpreal/blob/master/mpreal.h) in order to use the multi precision in addition to installing the dependencies in the environment.yml
