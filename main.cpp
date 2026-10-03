#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iomanip>

// =========================================================================
// STRUCTURES & ब्लू-प्रिंट्स ( Blueprints )
// =========================================================================

struct TradingTrackerNode {
    double actualVWAP = 0.0;       
    double externalForecast = 0.0; 
    double rawResidual = 0.0;      
    double waveletCleanErr = 0.0;  
    double correctedTarget = 0.0;  
};

struct ModelPerformanceMetrics {
    double sequenceLogLikelihood = 0.0;
    double modelMAE = 0.0;
    double modelR2Score = 0.0;
    double hurstExponent = 0.5;
    double phasorAlignedTarget = 0.0;
    double phasorMAE = 0.0;
    double bedrosianSeparationIndex = 1.0; 
    bool bedrosianSpectralBleedingAlert = false;
    std::string regimeStabilityAlert = "";
};

struct ExecutiveMetricsSummary {
    double snrdB = 0.0;
    double signalVar = 0.0;
    double noiseVar = 0.0;
    double scaleEnergy[6] = {0.0};
    int lowestNoiseScale = 1;
    double deltaPhase = 0.0;
    double stepInterventionDirection = 0.0;
    double currentL1 = 0.0;
    double stdL1 = 0.0;
    double correctedVal = 0.0;
    double fcastVal = 0.0;
    double scaleImpact[6] = {0.0};
    double scaleContrib[6] = {0.0};
    std::string scaleDirection[6];
    std::string algorithmicSignal = "";
    std::string microAlertText = "";
};

struct KonishiRoutingSchedule {
    double targetOrderSize = 100000.0;    
    double liquidityLagFactor = 0.0;     
    double optimalSlices[6] = {0.0};      
    std::string profileModeAlert = "";   
};

// =========================================================================
// NATIVE MATHEMATICAL ENGINE FUNCTIONS (Zero-Dependency)
// =========================================================================

double Native_Gamma_Lanczos(double x) {
    const double PI_VAL = 3.14159265358979323846;
    static const double p[] = {
        0.99999999999981,    676.520368121885,   -1259.1392167224,
        771.323428777653,   -176.615029162141,    12.5073432786869,
        -0.13857109526572,   9.98436957801957e-06, 1.50563273514931e-07
    };
    if (x < 0.5) return PI_VAL / (std::sin(PI_VAL * x) * Native_Gamma_Lanczos(1.0 - x));
    x -= 1.0;
    double agg = p[0];
    for (int j = 1; j <= 8; ++j) {
        agg += p[j] / (x + static_cast<double>(j));
    }
    double T = x + 7.5;
    return std::sqrt(2.0 * PI_VAL) * std::pow(T, x + 0.5) * std::exp(-T) * agg;
}

double Native_Skewed_T_Density(double res, double scaleSig, double nu, double lam) {
    const double PI_VAL = 3.14159265358979323846;
    double c = Native_Gamma_Lanczos((nu + 1.0) / 2.0) / (std::sqrt(PI_VAL * (nu - 2.0)) * Native_Gamma_Lanczos(nu / 2.0));
    double a = 4.0 * lam * c * ((nu - 2.0) / (nu - 1.0));
    double b = std::sqrt(1.0 + 3.0 * (lam * lam) - (a * a));
    double z = (res / (scaleSig + 1e-30) - a) / (b + 1e-30);
    if (z < 0.0) {
        return (b * c / scaleSig) * std::pow(1.0 + (z * z) / ((nu - 2.0) * ((1.0 - lam) * (1.0 - lam))), -(nu + 1.0) / 2.0);
    } else {
        return (b * c / scaleSig) * std::pow(1.0 + (z * z) / ((nu - 2.0) * ((1.0 + lam) * (1.0 + lam))), -(nu + 1.0) / 2.0);
    }
}

double Calculate_Hurst_Exponent(const std::vector<double>& observations) {
    size_t T = observations.size();
    size_t N = T - 1;
    std::vector<double> returns(N, 0.0);
    for (size_t i = 0; i < N; ++i) {
        if (observations[i] > 0.0) returns[i] = std::log(observations[i + 1] / observations[i]);
    }
    int sizes[] = {16, 32, 64, 128};
    double logN[4] = {0.0}, logRS[4] = {0.0};
    for (int i = 0; i < 4; ++i) {
        int subSize = sizes[i];
        int numSubs = N / subSize;
        double rsSum = 0.0;
        for (int j = 0; j < numSubs; ++j) {
            int idx = j * subSize;
            double sumRet = 0.0;
            for (int k = 0; k < subSize; ++k) sumRet += returns[idx + k];
            double avgRet = sumRet / subSize;
            double meanVar = 0.0;
            for (int k = 0; k < subSize; ++k) meanVar += std::pow(returns[idx + k] - avgRet, 2);
            double sampleStdDev = std::sqrt(meanVar / (subSize - 1));
            if (sampleStdDev < 0.00001) sampleStdDev = 0.00001;
            double cumSum = 0.0, minX = 0.0, maxX = 0.0;
            for (int k = 0; k < subSize; ++k) {
                cumSum += (returns[idx + k] - avgRet);
                if (cumSum > maxX) maxX = cumSum;
                if (cumSum < minX) minX = cumSum;
            }
            rsSum += ((maxX - minX) / sampleStdDev);
        }
        logN[i] = std::log(static_cast<double>(subSize));
        logRS[i] = std::log(rsSum / numSubs);
    }
    double logNMean = (logN[0] + logN[1] + logN[2] + logN[3]) / 4.0;
    double logRSMean = (logRS[0] + logRS[1] + logRS[2] + logRS[3]) / 4.0;
    double numNum = 0.0, denNum = 0.0;
    for (int i = 0; i < 4; ++i) {
        numNum += (logN[i] - logNMean) * (logRS[i] - logRSMean);
        denNum += std::pow(logN[i] - logNMean, 2);
    }
    return (denNum > 0.0) ? (numNum / denNum) : 0.5;
}

double Run_Bedrosian_Spectral_Check(const std::vector<double>& lowPassTrend, const std::vector<double>& highPassCycle, bool& alertTriggered) {
    size_t n = lowPassTrend.size();
    auto computeDFT = [](const std::vector<double>& signal) {
        size_t n = signal.size();
        std::vector<double> magnitudes(n / 2 + 1, 0.0);
        const double PI_VAL = 3.14159265358979323846;
        for (size_t k = 0; k <= n / 2; ++k) {
            double realPart = 0.0, imagPart = 0.0;
            for (size_t t = 0; t < n; ++t) {
                double angle = (2.0 * PI_VAL * k * t) / n;
                realPart += signal[t] * std::cos(angle);
                imagPart -= signal[t] * std::sin(angle);
            }
            magnitudes[k] = std::sqrt(realPart * realPart + imagPart * imagPart);
        }
        return magnitudes;
    };
    std::vector<double> magTrend = computeDFT(lowPassTrend);
    std::vector<double> magCycle = computeDFT(highPassCycle);
    double totalTrendPower = 0.0, totalCyclePower = 0.0, overlapBleeding = 0.0;
    for (size_t k = 0; k < magTrend.size(); ++k) {
        totalTrendPower += magTrend[k];
        totalCyclePower += magCycle[k];
        overlapBleeding += std::min(magTrend[k], magCycle[k]);
    }
    double totalSystemPower = (totalTrendPower + totalCyclePower + 1e-10);
    double separationIndex = 1.0 - (overlapBleeding / totalSystemPower);
    alertTriggered = (separationIndex < 0.82);
    return separationIndex;
}

// =========================================================================
// CORE INTEGRATED EXECUTION PIPELINE
// =========================================================================

void Execute_Unified_Trading_Engine(std::vector<TradingTrackerNode>& trackingBuffer, ModelPerformanceMetrics& metrics, ExecutiveMetricsSummary& report) {
    size_t T = trackingBuffer.size();
    if (T < 256) return;

    const double EPSILON_HMM = 1e-90;
    std::vector<double> obs(T);
    for (size_t i = 0; i < T; ++i) obs[i] = trackingBuffer[i].actualVWAP;

    // --- PHASE 1: APGARCH-HMM STEP ---
    std::vector<double> alpha1(T), alpha2(T), beta1(T), beta2(T), scaleFactors(T);
    std::vector<double> emission1(T), emission2(T), sig1(T), sig2(T), gamma1(T), gamma2(T);
    double pi1 = 0.5, pi2 = 0.5;
    double A11 = 0.9, A12 = 0.1, A21 = 0.1, A22 = 0.9;
    double c1 = obs[0] * 0.05, phi1 = 0.95, sigma1 = 1.0;
    double c2 = obs[0] * 0.1,  phi2 = 0.85, sigma2 = 2.0;
    double omega1 = sigma1 * 0.1,  alphaG1 = 0.08, betaG1 = 0.8,  gammaG1 = 0.25, deltaG1 = 1.4;
    double omega2 = sigma2 * 0.15, alphaG2 = 0.12, betaG2 = 0.75, gammaG2 = 0.35, deltaG2 = 1.4;
    double nu1 = 6.5, lam1 = -0.15, nu2 = 4.2, lam2 = -0.3;
    double currentLogLikelihood = 0.0, previousLogLikelihood = -1e30;

    for (size_t iter = 1; iter <= 40; ++iter) {
        emission1[0] = 1.0; emission2[0] = 1.0; sig1[0] = sigma1; sig2[0] = sigma2;
        for (size_t i = 1; i < T; ++i) {
            double res1 = obs[i] - (c1 + phi1 * obs[i - 1]);
            double res2 = obs[i] - (c2 + phi2 * obs[i - 1]);
            double shock1 = std::max(0.0, std::abs(res1) - gammaG1 * res1);
            double shock2 = std::max(0.0, std::abs(res2) - gammaG2 * res2);
            double pSig1 = std::max(0.0001, omega1 + alphaG1 * std::pow(shock1, deltaG1) + betaG1 * std::pow(sig1[i - 1], deltaG1));
            double pSig2 = std::max(0.0001, omega2 + alphaG2 * std::pow(shock2, deltaG2) + betaG2 * std::pow(sig2[i - 1], deltaG2));
            sig1[i] = std::pow(pSig1, 1.0 / deltaG1);
            sig2[i] = std::pow(pSig2, 1.0 / deltaG2);
            emission1[i] = std::max(EPSILON_HMM, Native_Skewed_T_Density(res1, sig1[i], nu1, lam1));
            emission2[i] = std::max(EPSILON_HMM, Native_Skewed_T_Density(res2, sig2[i], nu2, lam2));
        }
        alpha1[0] = pi1 * emission1[0]; alpha2[0] = pi2 * emission2[0];
        double denom = std::max(EPSILON_HMM, alpha1[0] + alpha2[0]);
        scaleFactors[0] = 1.0 / denom; alpha1[0] *= scaleFactors[0]; alpha2[0] *= scaleFactors[0];
        for (size_t t = 1; t < T; ++t) {
            alpha1[t] = (alpha1[t - 1] * A11 + alpha2[t - 1] * A21) * emission1[t];
            alpha2[t] = (alpha1[t - 1] * A12 + alpha2[t - 1] * A22) * emission2[t];
            denom = std::max(EPSILON_HMM, alpha1[t] + alpha2[t]);
            scaleFactors[t] = 1.0 / denom; alpha1[t] *= scaleFactors[t]; alpha2[t] *= scaleFactors[t];
        }
        beta1[T - 1] = scaleFactors[T - 1]; beta2[T - 1] = scaleFactors[T - 1];
        for (size_t t = T - 1; t > 0; --t) {
            size_t tStep = t - 1;
            beta1[tStep] = (A11 * emission1[t] * beta1[t] + A12 * emission2[t] * beta2[t]) * scaleFactors[tStep];
            beta2[tStep] = (A21 * emission1[t] * beta1[t] + A22 * emission2[t] * beta2[t]) * scaleFactors[tStep];
        }
        currentLogLikelihood = 0.0;
        for (size_t t = 1; t < T; ++t) currentLogLikelihood += std::log(1.0 / scaleFactors[t]);
        if (iter > 1 && (currentLogLikelihood - previousLogLikelihood) < 0.001) break;
        previousLogLikelihood = currentLogLikelihood;

        for (size_t t = 0; t < T; ++t) {
            double totalG = std::max(EPSILON_HMM, alpha1[t] * beta1[t] + alpha2[t] * beta2[t]);
            gamma1[t] = (alpha1[t] * beta1[t]) / totalG; gamma2[t] = (alpha2[t] * beta2[t]) / totalG;
        }
        double xi11 = 0, xi12 = 0, xi21 = 0, xi22 = 0, gSum1 = 0, gSum2 = 0;
        for (size_t t = 0; t < T - 1; ++t) {
            double jScale = std::max(EPSILON_HMM, alpha1[t] * A11 * emission1[t + 1] * beta1[t + 1] + alpha1[t] * A12 * emission2[t + 1] * beta2[t + 1] + alpha2[t] * A21 * emission1[t + 1] * beta1[t + 1] + alpha2[t] * A22 * emission2[t + 1] * beta2[t + 1]);
            xi11 += (alpha1[t] * A11 * emission1[t + 1] * beta1[t + 1]) / jScale;
            xi12 += (alpha1[t] * A12 * emission2[t + 1] * beta2[t + 1]) / jScale;
            xi21 += (alpha2[t] * A21 * emission1[t + 1] * beta1[t + 1]) / jScale;
            xi22 += (alpha2[t] * A22 * emission2[t + 1] * beta2[t + 1]) / jScale;
            gSum1 += gamma1[t]; gSum2 += gamma2[t];
        }
        gSum1 = std::max(EPSILON_HMM, gSum1); gSum2 = std::max(EPSILON_HMM, gSum2);
        A11 = xi11 / gSum1; A12 = xi12 / gSum1; A21 = xi21 / gSum2; A22 = xi22 / gSum2;

        double sY1=0, sX1=0, sXY1=0, sXX1=0, sG1=0, sY2=0, sX2=0, sXY2=0, sXX2=0, sG2=0;
        for (size_t t = 1; t < T; ++t) {
            sG1 += gamma1[t]; sY1 += gamma1[t]*obs[t]; sX1 += gamma1[t]*obs[t-1]; sXY1 += gamma1[t]*obs[t]*obs[t-1]; sXX1 += gamma1[t]*(obs[t-1]*obs[t-1]);
            sG2 += gamma2[t]; sY2 += gamma2[t]*obs[t]; sX2 += gamma2[t]*obs[t-1]; sXY2 += gamma2[t]*obs[t]*obs[t-1]; sXX2 += gamma2[t]*(obs[t-1]*obs[t-1]);
        }
        sG1 = std::max(EPSILON_HMM, sG1); sG2 = std::max(EPSILON_HMM, sG2);
        double d1 = sG1 * sXX1 - (sX1 * sX1); if (std::abs(d1) < 0.001) d1 = 0.001;
        phi1 = (sG1 * sXY1 - sX1 * sY1) / d1; c1 = (sY1 - phi1 * sX1) / sG1;
        double d2 = sG2 * sXX2 - (sX2 * sX2); if (std::abs(d2) < 0.001) d2 = 0.001;
        phi2 = (sG2 * sXY2 - sX2 * sY2) / d2; c2 = (sY2 - phi2 * sX2) / sG2;
        pi1 = gamma1[0]; pi2 = gamma2[0];
    }

    double x_hat1 = obs[T - 1], x_hat2 = obs[T - 1];
    double P_cov1 = std::pow(sig1[T - 1], 2), P_cov2 = std::pow(sig2[T - 1], 2);
    double R_noise = (sig1[T - 1] + sig2[T - 1]) * 0.5;
    trackingBuffer[0].externalForecast = obs[0];

    for (size_t t = 1; t < T; ++t) {
        double x_pred1 = c1 + phi1 * x_hat1; double x_pred2 = c2 + phi2 * x_hat2;
        double P_pred1 = (phi1 * P_cov1 * phi1) + std::pow(sig1[t], 2);
        double P_pred2 = (phi2 * P_cov2 * phi2) + std::pow(sig2[t], 2);
        double K1 = P_pred1 / (P_pred1 + R_noise); double K2 = P_pred2 / (P_pred2 + R_noise);
        x_hat1 = x_pred1 + K1 * (obs[t] - x_pred1); x_hat2 = x_pred2 + K2 * (obs[t] - x_pred2);
        P_cov1 = (1.0 - K1) * P_pred1; P_cov2 = (1.0 - K2) * P_pred2;
        double predP1 = (alpha1[t - 1] * A11) + (alpha2[t - 1] * A21);
        double predP2 = (alpha1[t - 1] * A12) + (alpha2[t - 1] * A22);
        trackingBuffer[t].externalForecast = (predP1 * x_pred1) + (predP2 * x_pred2);
    }

    // --- PHASE 2: WAVELET STRIP ---
    double h0 = (1.0 + std::sqrt(3.0)) / (4.0 * std::sqrt(2.0));
    double h1 = (3.0 + std::sqrt(3.0)) / (4.0 * std::sqrt(2.0));
    double h2 = (3.0 - std::sqrt(3.0)) / (4.0 * std::sqrt(2.0));
    double h3 = (1.0 - std::sqrt(3.0)) / (4.0 * std::sqrt(2.0));
    double g0 = h3, g1 = -h2, g2 = h1, g3 = -h0;

    std::vector<std::vector<double>> appMatrix(7, std::vector<double>(T, 0.0));
    std::vector<std::vector<double>> detMatrix(7, std::vector<double>(T, 0.0));
    std::vector<double> threshVectors(7, 0.0);
    std::vector<double> currentIn(T);

    std::vector<double> rawSquaredErrors(T);
    for (size_t i = 0; i < T; ++i) {
        trackingBuffer[i].rawResidual = 0.02 * (trackingBuffer[i].actualVWAP - trackingBuffer[i].externalForecast);
        currentIn[i] = trackingBuffer[i].rawResidual;
        rawSquaredErrors[i] = currentIn[i];
    }

    size_t currentLength = T;
    for (size_t lev = 1; lev <= 6; ++lev) {
        size_t halfLen = currentLength / 2;
        std::vector<double> tempApp(halfLen), tempDet(halfLen);
        for (size_t i = 0; i < halfLen; ++i) {
            size_t idx = 2 * i;
            size_t i0 = idx, i1 = (idx + 1) % currentLength, i2 = (idx + 2) % currentLength, i3 = (idx + 3) % currentLength;
            tempApp[i] = h0 * currentIn[i0] + h1 * currentIn[i1] + h2 * currentIn[i2] + h3 * currentIn[i3];
            tempDet[i] = g0 * currentIn[i0] + g1 * currentIn[i1] + g2 * currentIn[i2] + g3 * currentIn[i3];
        }
        size_t expand = T / halfLen;
        for (size_t i = 0; i < halfLen; ++i) {
            for (size_t j = 0; j < expand; ++j) {
                appMatrix[lev][i * expand + j] = tempApp[i];
                detMatrix[lev][i * expand + j] = tempDet[i];
            }
        }
        double sumD = 0.0;
        for (size_t i = 0; i < T; ++i) sumD += std::abs(detMatrix[lev][i]);
        threshVectors[lev] = sumD / T;
        currentIn.assign(tempApp.begin(), tempApp.end());
        currentLength = halfLen;
    }

    for (size_t i = 0; i < T; ++i) {
        for (size_t lev = 1; lev <= 6; ++lev) {
            double dVal = detMatrix[lev][i];
            double lambda = (lev == 1) ? threshVectors[lev] * 0.45 : threshVectors[lev] * (1.1 - (lev * 0.08));
            if (std::abs(dVal) <= lambda) detMatrix[lev][i] = 0.0;
            else detMatrix[lev][i] = dVal - ((lambda * lambda) / dVal);
        }
        trackingBuffer[i].waveletCleanErr = appMatrix[6][i] + detMatrix[1][i] + detMatrix[2][i] + detMatrix[3][i] + detMatrix[4][i] + detMatrix[5][i] + detMatrix[6][i];
    }

    // --- PHASE 3: BEDROSIAN & PHASOR GENERATOR ---
    std::vector<double> extractedTrends(T), residualCycles(T);
    for (size_t i = 0; i < T; ++i) {
        extractedTrends[i] = trackingBuffer[i].externalForecast;
        residualCycles[i] = trackingBuffer[i].waveletCleanErr;
    }
    metrics.bedrosianSeparationIndex = Run_Bedrosian_Spectral_Check(extractedTrends, residualCycles, metrics.bedrosianSpectralBleedingAlert);

    const double PI_VAL = 3.14159265358979323846;
    double hilbertTap[] = { -2.0/(3.0*PI_VAL), 0.0, -2.0/PI_VAL, 0.0, 2.0/PI_VAL, 0.0, 2.0/(3.0*PI_VAL) };
    std::vector<double> imagSignal(T, 0.0);

    for (size_t i = 0; i < T; ++i) {
        double convSum = 0.0;
        for (int j = -3; j <= 3; ++j) {
            long lIdx = static_cast<long>(i) + j;
            if (lIdx < 0) lIdx = -lIdx;
            else if (lIdx >= static_cast<long>(T)) lIdx = static_cast<long>(T) - (lIdx - static_cast<long>(T)) - 1;
            convSum += trackingBuffer[lIdx].waveletCleanErr * hilbertTap[j + 3];
        }
        imagSignal[i] = convSum;
    }

    double totalAbsErr = 0.0;
    double bleedingCorrectionFactor = metrics.bedrosianSpectralBleedingAlert ? 0.70 : 1.0;

    for (size_t i = 0; i < T; ++i) {
        double fcast = trackingBuffer[i].externalForecast;
        double actual = trackingBuffer[i].actualVWAP;
        double prevAct = (i > 0) ? trackingBuffer[i - 1].actualVWAP : actual;
        double adjBase = fcast;
        if (fcast < 5.0 && actual > 10.0) adjBase = prevAct * (1.0 + (fcast - std::floor(fcast)) * 0.01);

        double phase = std::atan2(imagSignal[i], trackingBuffer[i].waveletCleanErr);
        double scaleAdj = std::sqrt(std::abs(trackingBuffer[i].waveletCleanErr)) * std::cos(phase) * bleedingCorrectionFactor;

        if (std::abs(scaleAdj) > (actual * 0.5)) {
            double sgn = (scaleAdj > 0.0) ? 1.0 : ((scaleAdj < 0.0) ? -1.0 : 0.0);
            scaleAdj = sgn * std::log(1.0 + std::abs(scaleAdj)) * (actual * 0.01);
        }
        double corrected = adjBase + scaleAdj;
        if (corrected > (actual * 1.5) || corrected < (actual * 0.5)) corrected = adjBase;
        trackingBuffer[i].correctedTarget = corrected;
        totalAbsErr += std::abs(actual - corrected);
    }

    metrics.sequenceLogLikelihood = currentLogLikelihood;
    metrics.phasorMAE = totalAbsErr / T;
    metrics.phasorAlignedTarget = trackingBuffer[T - 1].correctedTarget;

    double sumAct = 0, ssTotal = 0, ssRes = 0;
    for (size_t i = 1; i < T; ++i) sumAct += trackingBuffer[i].actualVWAP;
    double meanAct = sumAct / (T - 1);
    metrics.modelMAE = 0.0;
    for (size_t i = 1; i < T; ++i) {
        metrics.modelMAE += std::abs(trackingBuffer[i].actualVWAP - trackingBuffer[i].externalForecast);
        ssTotal += std::pow(trackingBuffer[i].actualVWAP - meanAct, 2);
        ssRes += std::pow(trackingBuffer[i].actualVWAP - trackingBuffer[i].externalForecast, 2);
    }
    metrics.modelMAE /= (T - 1);
    metrics.modelR2Score = (ssTotal > 0.0) ? (1.0 - (ssRes / ssTotal)) : 0.0;
    metrics.hurstExponent = Calculate_Hurst_Exponent(obs);

    double finalPredP1 = (alpha1[T-1]*A11) + (alpha2[T-1]*A21);
    int nextDom = (finalPredP1 >= 0.5) ? 1 : 2;
    int curDom = (alpha1[T - 1] >= 0.5) ? 1 : 2;
    metrics.regimeStabilityAlert = (nextDom != curDom) ? "CRITICAL JUMP EXPECTED!" : "Stable (No Shift Predicted)";

    // --- PHASE 4: EXECUTIVE STATS PASS ---
    double sumSigVar = 0.0, sumNoiseVar = 0.0, totalEnergy = 0.0, sumL1 = 0.0;
    std::vector<double> l1Vector(T, 0.0);

    for (size_t i = 0; i < T; ++i) {
        double rawVal = rawSquaredErrors[i];
        double cleanVal = trackingBuffer[i].waveletCleanErr;
        double noiseVal = rawVal - cleanVal;
        sumSigVar += (cleanVal * cleanVal);
        sumNoiseVar += (noiseVal * noiseVal);

        for (size_t lev = 1; lev <= 6; ++lev) {
            double dVal = detMatrix[lev][i];
            report.scaleEnergy[lev-1] += (dVal * dVal);
            totalEnergy += (dVal * dVal);
            if (lev == 1) { l1Vector[i] = dVal; sumL1 += dVal; }
        }
    }
    double avgL1 = sumL1 / T; double sumSqL1 = 0.0;
    for (size_t i = 0; i < T; ++i) sumSqL1 += std::pow(l1Vector[i] - avgL1, 2);
    report.stdL1 = std::sqrt(sumSqL1 / (T - 1));
    if (report.stdL1 < 1e-15) report.stdL1 = 1e-15;
    report.currentL1 = l1Vector[T - 1];
    report.signalVar = sumSigVar / T; report.noiseVar = sumNoiseVar / T;
    if (report.signalVar < 1e-15) report.signalVar = 1e-15;
    if (report.noiseVar < 1e-15) report.noiseVar = 1e-15;
    report.snrdB = 10.0 * (std::log(report.signalVar / report.noiseVar) / std::log(10.0));

    int lowestNoiseScale = 1; double maxEnergyDensity = -1.0;
    if (totalEnergy > 1e-15) {
        for (size_t lev = 1; lev <= 6; ++lev) {
            report.scaleEnergy[lev-1] /= totalEnergy;
            if (report.scaleEnergy[lev-1] > maxEnergyDensity) {
                maxEnergyDensity = report.scaleEnergy[lev-1]; lowestNoiseScale = static_cast<int>(lev);
            }
        }
    } else {
        for (size_t lev = 1; lev <= 6; ++lev) report.scaleEnergy[lev-1] = 1.0 / 6.0;
        lowestNoiseScale = 1;
    }
    report.lowestNoiseScale = lowestNoiseScale;
    report.correctedVal = trackingBuffer[T - 1].correctedTarget;
    report.fcastVal = trackingBuffer[T - 1].externalForecast;
    report.stepInterventionDirection = detMatrix[lowestNoiseScale][T - 1];

    double phaseCurrent = std::abs(report.correctedVal - report.fcastVal);
    if (T > 1) {
        double phasePrevious = std::abs(trackingBuffer[T - 2].correctedTarget - trackingBuffer[T - 2].externalForecast);
        report.deltaPhase = phaseCurrent - phasePrevious;
    } else report.deltaPhase = 0.0;

    if (report.stepInterventionDirection > 0.0 && report.correctedVal > report.fcastVal) report.algorithmicSignal = "ALPHA DIRECTIONAL BUY - ACCELERATING FORCE";
    else if (report.stepInterventionDirection < 0.0 && report.correctedVal < report.fcastVal) report.algorithmicSignal = "ALPHA DIRECTIONAL SELL - EXHAUSTION IMPULSE";
    else report.algorithmicSignal = "CONSOLIDATION - HOLD SPREAD TARGET";

    if (std::abs(report.currentL1) > (2.5 * report.stdL1)) report.microAlertText = (report.currentL1 > 0.0) ? "POSITIVE MICRO-IMPULSE SHOCK: Sudden upward liquidity surge!" : "NEGATIVE MICRO-IMPULSE SHOCK: Sudden downward liquidity drop!";
    else if (std::abs(report.currentL1) < (0.1 * report.stdL1) && std::abs(report.currentL1) > 1e-9) report.microAlertText = "MICRO-LIQUIDITY COMPRESSION: Breakout imminent.";
    else report.microAlertText = "Normal Noise Floor Environment.";

    double totalAbsImpact = 0.0;
    for (size_t lev = 1; lev <= 6; ++lev) {
        report.scaleImpact[lev-1] = detMatrix[lev][T - 1]; totalAbsImpact += std::abs(report.scaleImpact[lev-1]);
    }
    if (totalAbsImpact > 1e-15) {
        for (size_t lev = 1; lev <= 6; ++lev) {
            report.scaleContrib[lev-1] = std::abs(report.scaleImpact[lev-1]) / totalAbsImpact;
            report.scaleDirection[lev-1] = (report.scaleImpact[lev-1] > 0.0) ? "[+] BULLISH" : ((report.scaleImpact[lev-1] < 0.0) ? "[-] BEARISH" : "[ ] NEUTRAL");
        }
    } else {
        for (size_t lev = 1; lev <= 6; ++lev) { report.scaleContrib[lev-1] = 0.0; report.scaleDirection[lev-1] = "[ ] NEUTRAL"; }
    }
}

// --- PHASE 5: KONISHI EXECUTION LAYER ---
void Compute_Konishi_Optimal_Execution_Slices(double snrdB, double l1Impulse, KonishiRoutingSchedule& schedule) {
    double baselineVolumeProfile[] = { 0.30, 0.15, 0.08, 0.07, 0.15, 0.25 };
    if (snrdB > 0.0) schedule.liquidityLagFactor = std::abs(l1Impulse) * (1.0 / (snrdB + 0.1)) * 0.5;
    else schedule.liquidityLagFactor = std::abs(l1Impulse) * 1.5;

    if (schedule.liquidityLagFactor > 0.5) schedule.liquidityLagFactor = 0.5;
    if (schedule.liquidityLagFactor < 0.0) schedule.liquidityLagFactor = 0.0;
    schedule.profileModeAlert = (schedule.liquidityLagFactor > 0.1) ? "[RISK SHIFT ACTIVE]" : "[Standard Volatility Profile]";

    double totalOptimalVolume = 0.0;
    for (size_t lev = 1; lev <= 6; ++lev) {
        size_t idx = lev - 1;
        if (lev == 1 || lev == 6) schedule.optimalSlices[idx] = baselineVolumeProfile[idx] * (1.0 - schedule.liquidityLagFactor);
        else if (lev == 2 || lev == 5) schedule.optimalSlices[idx] = baselineVolumeProfile[idx] * (1.0 + (schedule.liquidityLagFactor * 1.2));
        else schedule.optimalSlices[idx] = baselineVolumeProfile[idx];
        totalOptimalVolume += schedule.optimalSlices[idx];
    }
    if (totalOptimalVolume < 1e-15) totalOptimalVolume = 1e-15;
    for (size_t idx = 0; idx < 6; ++idx) schedule.optimalSlices[idx] = (schedule.optimalSlices[idx] / totalOptimalVolume) * schedule.targetOrderSize;
}

// =========================================================================
// I/O WORKSPACE PARSER ENGINES
// =========================================================================

bool Load_VWAP_CSV_Feed(const std::string& filename, std::vector<TradingTrackerNode>& marketFeed) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;
    marketFeed.clear(); std::string line; bool isHeader = true;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        //  NEW REPAIRED CODE
if (line.empty()) continue;
if (line.back() == '\r' || line.back() == '\n') line.pop_back();
std::stringstream ss(line); std::string cell;

        if (isHeader) {
            if (!line.empty() && std::isdigit(line[0])) isHeader = false;
            else { isHeader = false; continue; }
        }
        if (std::getline(ss, cell, ',')) {
            try { TradingTrackerNode node; node.actualVWAP = std::stod(cell); marketFeed.push_back(node); }
            catch (...) { continue; }
        }
    }
    file.close();
    if (marketFeed.size() < 256) return false;
    if (marketFeed.size() > 256) marketFeed.erase(marketFeed.begin(), marketFeed.end() - 256);
    return true;
}

bool Export_Pipeline_Results(const std::string& filename, const std::vector<TradingTrackerNode>& metricsGrid) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) return false;
    outFile << "Actual_VWAP,HMM_Forecast,Raw_Residual,Wavelet_Clean_Cycle,Phasor_Corrected_Target\n";
    outFile << std::fixed << std::setprecision(5);
    for (const auto& row : metricsGrid) {
        outFile << row.actualVWAP << "," << row.externalForecast << "," << row.rawResidual << "," << row.waveletCleanErr << "," << row.correctedTarget << "\n";
    }
    outFile.close();
    return true;
}

// =========================================================================
// RUN TIME EXECUTION HARNESS
// =========================================================================

int main() {
    std::string inputPath = "input_vwap.csv";
    std::string outputPath = "output_predictions.csv";
    std::vector<TradingTrackerNode> marketFeed;

    std::cout << ">> Initializing Pipeline Engine...\n";
    if (!Load_VWAP_CSV_Feed(inputPath, marketFeed)) {
        std::cout << ">> Warning: input_vwap.csv not found or insufficient. Generating artificial vector (256 nodes)...\n";
        marketFeed.resize(256);
        for (size_t i = 0; i < 256; ++i) marketFeed[i].actualVWAP = 120.0 + (i * 0.02) + (std::sin(i * 0.1) * 3.0);
        std::ofstream mockIn(inputPath);
        for (size_t i = 0; i < 256; ++i) mockIn << marketFeed[i].actualVWAP << "\n";
        mockIn.close();
    }

    ModelPerformanceMetrics report;
    ExecutiveMetricsSummary execSummary;
    KonishiRoutingSchedule schedule;

    std::cout << ">> Processing Multi-Resolution Data Array Stack...\n";
    Execute_Unified_Trading_Engine(marketFeed, report, execSummary);
    Compute_Konishi_Optimal_Execution_Slices(execSummary.snrdB, execSummary.currentL1, schedule);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n=======================================================================\n";
    std::cout << "             MULTI-RESOLUTION ENGINE CORE DIAGNOSTIC REPORT            \n";
    std::cout << "=======================================================================\n";
    std::cout << "1. WAVELET FILTERING EFFICIENCY METRICS\n";
    std::cout << "   Signal-to-Noise Ratio (SNR)        : " << execSummary.snrdB << " dB\n";
    std::cout << "   Model Forecast R2 Variance  (R2)   : " << report.modelR2Score << "\n";
    std::cout << "   Hurst Fractality Exponent   (H)    : " << report.hurstExponent << "\n\n";
    std::cout << "2. BEDROSIAN VALIDATION PROFILE\n";
    std::cout << "   Bedrosian Separation Score         : " << report.bedrosianSeparationIndex << " (" 
              << (report.bedrosianSpectralBleedingAlert ? "BLEEDING ACTIVE" : "CLEAN INDEPENDENCE") << ")\n\n";
    std::cout << "3. ADVANCED PHASE KINETICS & ALGORITHMIC SIGNALS\n";
    std::cout << "   Phasor Aligned Output Target (T+1) : " << report.phasorAlignedTarget << "\n";
    std::cout << "   Regime Change Notification         : " << report.regimeStabilityAlert << "\n";
    std::cout << "   ALGORITHMIC TRADING SIGNAL         : " << execSummary.algorithmicSignal << "\n\n";
    std::cout << "4. HIGH-FREQUENCY ALERT ENGINE\n";
    std::cout << "   L1 Noise Shock Alert Level         : " << execSummary.microAlertText << "\n";

    std::cout << "\n=======================================================================\n";
    std::cout << "       6. MR. HIZURU KONISHI OPTIMAL LIQUIDITY-LAG ROUTING SCHEDULE    \n";
    std::cout << "=======================================================================\n";
    std::cout << "   Calculated Liquidity Lag Factor : " << (schedule.liquidityLagFactor * 100.0) << "%\n";
    std::cout << "   Operational Profile Tracking    : " << schedule.profileModeAlert << "\n\n";
    const char* labels[] = { "Interval T1 (Open Peak) ", "Interval T2 (Post-Peak) ", "Interval T3 (Midday Lull)", "Interval T4 (Early Aft) ", "Interval T5 (Pre-Close) ", "Interval T6 (Close Peak) " };
    for (size_t i = 0; i < 6; ++i) {
        std::cout << "   " << labels[i] << " : " << std::setw(6) << std::fixed << std::setprecision(0) << schedule.optimalSlices[i] << " shares\n";
    }
    std::cout << "=======================================================================\n";

    Export_Pipeline_Results(outputPath, marketFeed);
    return 0;
}
