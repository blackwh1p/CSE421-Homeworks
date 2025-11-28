/*
 * hdr_feature_extraction.c
 * - Fixed Row-Major indexing
 * - Implemented Double Precision
 * - Implemented Log Transform for Hu Moments
 */

#include "hdr_feature_extraction.h"
#include <math.h>
#include <string.h>

// Helper for integer power to avoid expensive powf() and maintain precision
static double fast_pow(double base, int exp)
{
    if (exp == 0) return 1.0;
    if (exp == 1) return base;
    if (exp == 2) return base * base;
    if (exp == 3) return base * base * base;
    return 1.0; 
}

int8_t hdr_calculate_moments(IMAGE_HandleTypeDef * img, HDR_FtrExtOutput *output)
{
    if (img->format != IMAGE_FORMAT_GRAYSCALE)
    {
        return -1;
    }

    // Use double for intermediate calculations to prevent precision loss
    double temp_moments[4][4]; 
    memset(temp_moments, 0, sizeof(temp_moments));

    // --- 1. Calculate Raw Moments (M_pq) ---
    // Iterate Row-Major (Y then X) to match Python/OpenCV memory layout
    for(uint32_t y = 0; y < img->height; y++) 
    {
        for (uint32_t x = 0; x < img->width; x++) 
        {
            uint8_t pixel = img->pData[y * img->width + x];
            
            // Thresholding: Matches Python cv2.threshold(128, 255, THRESH_BINARY)
            // If > 128, treat as 1.0 (Normalized binary mass)
            double pixel_val = (pixel > 128) ? 1.0 : 0.0;

            if (pixel_val > 0.0) 
            {
                for(int i = 0; i < 4; i++)
                {
                    for(int j = 0; j < 4 - i; j++)
                    {
                        temp_moments[i][j] += fast_pow((double)x, i) * fast_pow((double)y, j) * pixel_val;
                    }
                }
            }
        }
    }

    // --- 2. Calculate Centroid ---
    if (temp_moments[0][0] == 0.0) return -1; // Empty image

    double centroid_x = temp_moments[1][0] / temp_moments[0][0];
    double centroid_y = temp_moments[0][1] / temp_moments[0][0];

    // --- 3. Calculate Central Moments (mu_pq) ---
    // Using temporary doubles
    double mu[4][4];
    memset(mu, 0, sizeof(mu));

    mu[0][0] = temp_moments[0][0];
    
    mu[2][0] = temp_moments[2][0] - centroid_x * temp_moments[1][0];
    mu[0][2] = temp_moments[0][2] - centroid_y * temp_moments[0][1];
    mu[1][1] = temp_moments[1][1] - centroid_x * temp_moments[0][1];

    mu[3][0] = temp_moments[3][0] - 3 * centroid_x * temp_moments[2][0] + 
               2 * fast_pow(centroid_x, 2) * temp_moments[1][0];
               
    mu[0][3] = temp_moments[0][3] - 3 * centroid_y * temp_moments[0][2] + 
               2 * fast_pow(centroid_y, 2) * temp_moments[0][1];

    mu[2][1] = temp_moments[2][1] - 2 * centroid_x * temp_moments[1][1] - 
               centroid_y * temp_moments[2][0] + 
               2 * fast_pow(centroid_x, 2) * temp_moments[0][1];

    mu[1][2] = temp_moments[1][2] - 2 * centroid_y * temp_moments[1][1] - 
               centroid_x * temp_moments[0][2] + 
               2 * fast_pow(centroid_y, 2) * temp_moments[1][0];

    // --- 4. Calculate Scale Invariant Moments (nu_pq) ---
    double area = mu[0][0];
    double norm_factor_2 = area * area; 
    double norm_factor_3 = norm_factor_2 * sqrt(area);

    // Store in output struct (cast to float is safe here)
    output->nu[2][0] = (float)(mu[2][0] / norm_factor_2);
    output->nu[0][2] = (float)(mu[0][2] / norm_factor_2);
    output->nu[1][1] = (float)(mu[1][1] / norm_factor_2);

    output->nu[3][0] = (float)(mu[3][0] / norm_factor_3);
    output->nu[0][3] = (float)(mu[0][3] / norm_factor_3);
    output->nu[2][1] = (float)(mu[2][1] / norm_factor_3);
    output->nu[1][2] = (float)(mu[1][2] / norm_factor_3);

    return 0;
}

void hdr_calculate_hu_moments(HDR_FtrExtOutput *output)
{
    // Use doubles for calculation
    double n20 = (double)output->nu[2][0];
    double n02 = (double)output->nu[0][2];
    double n11 = (double)output->nu[1][1];
    double n30 = (double)output->nu[3][0];
    double n12 = (double)output->nu[1][2];
    double n21 = (double)output->nu[2][1];
    double n03 = (double)output->nu[0][3];

    double h[7];

    // Standard Hu Moment Formulas
    h[0] = n20 + n02;

    h[1] = fast_pow(n20 - n02, 2) + 4 * fast_pow(n11, 2);

    h[2] = fast_pow(n30 - 3 * n12, 2) + fast_pow(3 * n21 - n03, 2);

    h[3] = fast_pow(n30 + n12, 2) + fast_pow(n21 + n03, 2);

    h[4] = (n30 - 3 * n12) * (n30 + n12) * 
           (fast_pow(n30 + n12, 2) - 3 * fast_pow(n21 + n03, 2)) + 
           (3 * n21 - n03) * (n21 + n03) * 
           (3 * fast_pow(n30 + n12, 2) - fast_pow(n21 + n03, 2));

    h[5] = (n20 - n02) * (fast_pow(n30 + n12, 2) - fast_pow(n21 + n03, 2)) + 
           4 * n11 * (n30 + n12) * (n21 + n03);

    h[6] = (3 * n21 - n03) * (n30 + n12) * 
           (fast_pow(n30 + n12, 2) - 3 * fast_pow(n21 + n03, 2)) - 
           (n30 - 3 * n12) * (n21 + n03) * 
           (3 * fast_pow(n30 + n12, 2) - fast_pow(n21 + n03, 2));

    // --- LOG TRANSFORM ---
    // This is crucial for scaling features to a range the Decision Tree can handle reliably.
    // Formula: -1 * sign(h) * log10(abs(h))
    for (int i = 0; i < 7; i++) 
    {
        if (h[i] != 0.0) 
        {
            double val = h[i];
            double sign = (val > 0.0) ? 1.0 : -1.0;
            // Use log10 of absolute value, flip sign to keep order
            h[i] = -1.0 * sign * log10(fabs(val));
        } 
        else 
        {
            h[i] = 0.0;
        }
        
        // Assign final transformed feature to output struct
        output->hu_moments[i] = (float)h[i];
    }
}