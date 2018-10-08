/* 
 * A linear regression model of vote share changes with intercepts and slopes
 * varying by region. The model generates predicted shares and a log likelihood
 * for each observation
 */

data {
  int<lower=1> N;
  int<lower=1> P;  // number of predictors
  int<lower=1> R;  // number of regions
  matrix[N, P] x;
  int<lower=1,upper=R> region[N];
  vector[N] y;     // change in vote share
}
transformed data {
  matrix[N, P] x_rescaled;
  for (p in 1:P){
    x_rescaled[,p] =
      sd(x[, p]) > 0 ?
      (x[,p] - mean(x[,p])) / sd(x[,p]) :
      x[, p];
  }
}
parameters {
  matrix[P, R] z_b; 
  vector[P] mu;
  real<lower=0> sigma;
  real nu;
  vector<lower=0>[P] tau;
}
transformed parameters {
  matrix[P, R] b;                      // slopes - one for each predictor/region
  for (p in 1:P)
    b[p] = mu[p] + z_b[p] * tau[p];
}
model {
  for (n in 1:N){
    y[n] ~ student_t(nu, x_rescaled[n] * b[, region[n]], sigma);
  }
  to_vector(z_b) ~ normal(0, 1);
  mu ~ normal(0, 1);
  tau ~ normal(0, 1);
  sigma ~ normal(0, 1);
  nu ~ gamma(2, 0.1);
}
generated quantities {
  vector[N] y_tilde;
  vector[N] log_lik;
  for (n in 1:N){
    y_tilde[n] = student_t_rng(nu, x_rescaled[n] * b[, region[n]], sigma);
    log_lik[n] = student_t_lpdf(y[n] | nu, x_rescaled[n] * b[, region[n]], sigma);
  }
}
