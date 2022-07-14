```python
from fastbook import *
from fastai.vision.widgets import *
```


```python
key = os.environ.get('AZURE_SEARCH_KEY', '079c978071fc4b018cbc369ff9b8fe02')
```


```python
results = search_images_bing(key, 'amy adams', 'isla fisher')
ims = results.attrgot('contentUrl')
len(ims)


```




    150




```python
ims[50]
```




    'http://celebsla.com/wp-content/uploads/2018/12/amy-adams-attends-vice-world-premiere-in-la-12-11-2018-3.jpg'




```python
dest = 'images/adamsfisher.jpg'
download_url(ims[0], dest)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>





<div>
  <progress value='655360' class='' max='654620' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.11% [655360/654620 00:00<00:00]
</div>






    Path('images/adamsfisher.jpg')




```python
im = Image.open(dest)
im.to_thumb(128,128)
```




    
![png](output_5_0.png)
    




```python
fis_ad = 'isla fisher', 'amy adams'
path = Path('fisheradams')
```


```python
if not path.exists():
    path.mkdir()
    for o in fis_ad:
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        results = search_images_bing(key, f'{o} fis_ad')
        download_images(dest, urls=results.attrgot('contentUrl'))
```


```python
fns = get_image_files(path)
```


```python
fns
```




    (#269) [Path('fisheradams/isla fisher/9abe734f-b58b-420f-aab8-869afde074e2.jpg'),Path('fisheradams/isla fisher/05806de3-ad53-40f9-a952-6b1814ffab82.jpg'),Path('fisheradams/isla fisher/e10ec216-ac56-46df-8124-1fedd4125835.jpg'),Path('fisheradams/isla fisher/42b4f1ec-0cf9-403c-9cd7-022481ca6c8e.jpg'),Path('fisheradams/isla fisher/a2fae0a2-b78c-4497-9199-98ae5a8aacd5.jpg'),Path('fisheradams/isla fisher/0ca104e7-d10d-45cb-b647-f659c83c7914.jpg'),Path('fisheradams/isla fisher/5a5713d0-e07f-464f-87a5-55e55d130e77.jpg'),Path('fisheradams/isla fisher/5635e29a-d1cb-41fc-9b33-ab24080a50ec.jpg'),Path('fisheradams/isla fisher/e2ee6810-e1a2-40de-a8a4-9ae9c4cb6414.jpg'),Path('fisheradams/isla fisher/b434aa8c-fa2c-4188-8630-5aa75322ec43.jpg')...]




```python
failed = verify_images(fns)
```


```python
failed
```




    (#0) []




```python
failed.map(Path.unlink);
```


```python
fisherORadams = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))
```


```python
dls = fisherORadams.dataloaders(path)
```


```python
dls.valid.show_batch(max_n=16, nrows=4)
```


    
![png](output_15_0.png)
    



```python
fisherORadams = fisherORadams.new(item_tfms=Resize(128, ResizeMethod.Pad))
dls = fisherORadams.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)
```


    
![png](output_16_0.png)
    



```python
fisherORadams = fisherORadams.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))
dls = fisherORadams.dataloaders(path)
dls.train.show_batch(max_n=8, nrows=2, unique=True)
```


    
![png](output_17_0.png)
    



```python
fisherORadams = fisherORadams.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())
dls = fisherORadams.dataloaders(path)
```


```python
dls.show_batch(max_n=8, nrows=2, unique=True)
```


    
![png](output_19_0.png)
    



```python
learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(15)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.376135</td>
      <td>0.966424</td>
      <td>0.339623</td>
      <td>00:05</td>
    </tr>
  </tbody>
</table>




<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.018136</td>
      <td>0.774253</td>
      <td>0.264151</td>
      <td>00:05</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.943862</td>
      <td>0.684239</td>
      <td>0.264151</td>
      <td>00:05</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.814111</td>
      <td>0.699748</td>
      <td>0.283019</td>
      <td>00:05</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.715987</td>
      <td>0.657071</td>
      <td>0.169811</td>
      <td>00:05</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.658370</td>
      <td>0.651735</td>
      <td>0.226415</td>
      <td>00:05</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.580376</td>
      <td>0.634504</td>
      <td>0.188679</td>
      <td>00:05</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.517035</td>
      <td>0.637869</td>
      <td>0.169811</td>
      <td>00:05</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.446339</td>
      <td>0.676268</td>
      <td>0.207547</td>
      <td>00:05</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.398060</td>
      <td>0.695591</td>
      <td>0.188679</td>
      <td>00:05</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.363166</td>
      <td>0.708805</td>
      <td>0.169811</td>
      <td>00:05</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.325187</td>
      <td>0.708384</td>
      <td>0.150943</td>
      <td>00:05</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.299447</td>
      <td>0.685885</td>
      <td>0.150943</td>
      <td>00:05</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.270137</td>
      <td>0.661883</td>
      <td>0.150943</td>
      <td>00:05</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.248463</td>
      <td>0.663270</td>
      <td>0.150943</td>
      <td>00:05</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.232983</td>
      <td>0.666363</td>
      <td>0.150943</td>
      <td>00:05</td>
    </tr>
  </tbody>
</table>



```python
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>








    
![png](output_21_4.png)
    



```python
interp.plot_top_losses(10, nrows=2)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>








    
![png](output_22_2.png)
    



```python
cleaner = ImageClassifierCleaner(learn)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>








```python
cleaner
```


    VBox(children=(Dropdown(options=('amy adams', 'isla fisher'), value='amy adams'), Dropdown(options=('Train', '…



```python
for idx in cleaner.delete(): cleaner.fns[idx].unlink()
```


```python
learn.export()
```


```python
path = Path()
path.ls(file_exts='.pkl')
```




    (#1) [Path('export.pkl')]




```python
learn_inf = load_learner(path/'export.pkl')
```


```python
btn_upload = widgets.FileUpload()
btn_upload
```


    FileUpload(value={}, description='Upload')



```python
img = PILImage.create(btn_upload.data[-1])
                                    
```


```python
out_pl = widgets.Output()
out_pl.clear_output()
with out_pl: display(img.to_thumb(128,128))
out_pl
```


    Output()



```python
pred,pred_idx,probs = learn_inf.predict(img)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>








```python
lbl_pred = widgets.Label()
lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
lbl_pred
```


    Label(value='Prediction: isla fisher; Probability: 0.9765')



```python
btn_run = widgets.Button(description='Classify')
btn_run
```


    Button(description='Classify', style=ButtonStyle())



```python
def on_click_classify(change):
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'

btn_run.on_click(on_click_classify)
```


```python
btn_upload = widgets.FileUpload()

```


```python
VBox([widgets.Label('Select your favorite oft-confused redheaded movie star!'),
                    btn_upload, btn_run, out_pl, lbl_pred])
```


    VBox(children=(Label(value='Select your favorite oft-confused redheaded movie star!'), FileUpload(value={'amy-…



```python
!pip install voila
!jupyter serverextention enable --sys-prefix voila
```


```python
btn_upload = widgets.FileUpload()
btn_upload
```


    FileUpload(value={}, description='Upload')

