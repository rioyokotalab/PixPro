# Dataset prepareration
## BDD100K dataset download and create image from movie
<data_path> is place of dataset which you determined  
### requiment
- `parallel`
- `aria2c` or `wget`
- `ffmpeg`
### create dataset using multiprocessing
if you want to create quickely, run job by multiprocessing  
<!-- (required command `ybatch` or `qsub` for job command)  -->
<!-- https://github.com/rioyokotalab/video-representation-learning/tree/main/scripts -->
1. download dataset

    ```shell script
    bash get_data/download_videos.sh <data_path>
    ```

2. unzip dataset

    ```shell script
    bash get_data/unzip_videos.sh <data_path>
    ```

3. create directory for images which you use in training

    ```shell script
    bash get_data/mkdir_train_val_img.sh <data_path>
    ```

4. finally, create images using multiprocessing

    Ex. create 1900/process using 37 process(machine)

    1st node  
    ```shell script
    bash get_data/create_img.sh <data_path> 1 1900
    ```
    2nd node  
    ```shell script
    bash get_data/create_img.sh <data_path> 1901 1900
    ```
    :  
    n-th node
    ```shell script
    bash get_data/create_img.sh <data_path> (n-1)*1900+1 1900
    ```
    :  
    37th node  
    ```shell script
    bash get_data/create_img.sh <data_path> 6840 1900
    ```


  <!-- 1. create job scripts for multiprocessing

      ```
      bash job_sh/gen_job_sh/gen_create_trainval_img_job.sh <data_path> 
      ```

  2. run job by using job scripts which you created above command

      ```
      bash job_sh/sub_job_sh/train_val_gen_img_job_sub.sh
      ``` -->

### create dataset using singleprocess
if you don't mind time to create dataset, run following command

```
bash process_bdd.sh <data_path>
```

