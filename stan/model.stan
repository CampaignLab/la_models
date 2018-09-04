/* 
 * A linear regression model of vote shares with intercepts and slopes varying
 * by region. The model generates predicted shares and a log likelihood for each
 * observation
 */

data {
  int<lower=1> N;
  int<lower=1> M;  // number of predictors
  int<lower=1> R;  // number of regions
  matrix[N, M] x;
  int<lower=1,upper=R> region[N];
  vector[N] y;
}
transformed data {
  matrix[N, M] x_std;
  for (m in 1:M){
    x_std[,m] = (x[,m] - mean(x[,m])) / sd(x[,m]);
  }
}
parameters {
  matrix[M, R] b_z;     // regression coefficients
  vector[R] a_z;    // region-specific intercept
  real mu_a;
  vector[M] mu_b;
  real<lower=0> sigma_y;
  real<lower=0> sigma_a;
  vector<lower=0>[M] sigma_b;
}
transformed parameters {
  matrix[M, R] b;
  vector[R] a = mu_a + a_z * sigma_a;
  for (m in 1:M){
    b[m] = mu_b[m] + b_z[m] * sigma_b[m];
  }
}
model {
  for (n in 1:N){
    y[n] ~ normal(a[region[n]] + x_std[n] * b[, region[n]], sigma_y);
  }
  a_z ~ normal(0, 1);
  to_vector(b_z) ~ normal(0, 1);
  mu_a ~ normal(0, 1);
  mu_b ~ normal(0, 1);
  sigma_a ~ normal(0, 1);
  sigma_b ~ normal(0, 1);
  sigma_y ~ normal(0, 1);
}
generated quantities {
  vector[N] y_tilde;
  vector[N] log_lik;
  for (n in 1:N){
    y_tilde[n] = normal_rng(a[region[n]] + x_std[n] * b[, region[n]], sigma_y);
    log_lik[n] = normal_lpdf(y[n] | a[region[n]] + x_std[n] * b[, region[n]], sigma_y);
  }
}
