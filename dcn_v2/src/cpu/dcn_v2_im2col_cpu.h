#ifndef DCN_V2_IM2COL_CPU
#define DCN_V2_IM2COL_CPU

#ifdef __cplusplus
extern "C"
{
#endif

  void modulated_deformable_im2col_cpu(
    const float *data_im, const float *data_offset, const float *data_mask,
    const int batch_size, const int channels, const int height_im, const int width_im,
    const int height_col, const int width_col, const int kernel_h, const int kenerl_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int deformable_group, float *data_col);

  void modulated_deformable_col2im_cpu(
    const float *data_col, const float *data_offset, const float *data_mask,
    const int batch_size, const int channels, const int height_im, const int width_im,
    const int height_col, const int width_col, const int kernel_h, const int kenerl_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int deformable_group, float *grad_im);

  void modulated_deformable_col2im_coord_cpu(
    const float *data_col, const float *data_im, const float *data_offset, const float *data_mask,
    const int batch_size, const int channels, const int height_im, const int width_im,
    const int height_col, const int width_col, const int kernel_h, const int kenerl_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int deformable_group,
    float *grad_offset, float *grad_mask);

#ifdef __cplusplus
}
#endif

#endif
