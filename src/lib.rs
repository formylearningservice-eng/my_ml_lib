use std::slice;

// 学習結果をJulia側に返すための構造体
#[repr(C)]
pub struct LinearRegressionResult {
    pub slope: f64,     // 傾き (θ1)
    pub intercept: f64, // 切片 (θ0)
}

/// 単回帰分析の学習関数
/// 
/// # Safety
/// x_ptr と y_ptr は有効なメモリ領域を指している必要があります。
#[no_mangle]
pub unsafe extern "C" fn train_simple_linear_regression(
    x_ptr: *const f64,
    y_ptr: *const f64,
    len: usize,
) -> LinearRegressionResult {
    if len == 0 {
        return LinearRegressionResult { slope: 0.0, intercept: 0.0 };
    }

    // Juliaから渡されたポインタをRustのスライスに安全に変換（ゼロコピー）
    let x = slice::from_raw_parts(x_ptr, len);
    let y = slice::from_raw_parts(y_ptr, len);

    // 平均の計算
    let x_mean = x.iter().sum::<f64>() / (len as f64);
    let y_mean = y.iter().sum::<f64>() / (len as f64);

    // 共分散と分散の計算
    let mut numerator = 0.0;   // Σ(x - x_mean)(y - y_mean)
    let mut denominator = 0.0; // Σ(x - x_mean)^2

    for i in 0..len {
        let x_diff = x[i] - x_mean;
        let y_diff = y[i] - y_mean;
        numerator += x_diff * y_diff;
        denominator += x_diff * x_diff;
    }

    let slope = if denominator != 0.0 { numerator / denominator } else { 0.0 };
    let intercept = y_mean - (slope * x_mean);

    LinearRegressionResult { slope, intercept }
}

/// 予測を行う関数
#[no_mangle]
pub unsafe extern "C" fn predict_simple_linear_regression(
    x_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    slope: f64,
    intercept: f64,
) {
    let x = slice::from_raw_parts(x_ptr, len);
    let out = slice::from_raw_parts_mut(out_ptr, len);

    for i in 0..len {
        out[i] = slope * x[i] + intercept;
    }
}
