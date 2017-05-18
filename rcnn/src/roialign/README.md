## My implement or roialign
**NOTE**:ROI align has two step:
- first step:For all regions in feature map,do bilinear interpolation over all of this
- second step:do max/vag pooling
we only implement the first step,you can use the pooling operation of mxnet to do the next step like this:
```
data = mx.sym.Variable('data')
rois = mx.sym.Variable('rois')
pool = mx.sym.ROIAlign(data=data,rois=rois,aligned_heigh=(7,7),spatial_scale=0.25)
pool = mx.sym.pooling(data=pool,kernel=(2,2,),stride=1,pool_type='max')
```

### how to use:
copy all files to `/mxnet/src/operator`, and `make clean`,then make follow mxnet install tutorial
