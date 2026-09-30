//! 光线追迹终止分类（诊断与验收用，禁止与真实星空混淆）。

use std::collections::HashMap;
use std::fmt;

/// 单条光线积分如何结束。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RayTermination {
    /// 落入事件视界内。
    Captured,
    /// 到达逃逸半径，方向有效。
    Escaped,
    /// 穿过薄盘（可多次，见 `disk_crossings`）。
    DiskCrossing,
    /// 超出 `max_steps` 预算。
    StepBudgetExceeded,
    /// 半径或四速度出现非有限值。
    NonFiniteState,
    /// 无法构造零测地线初值。
    InitFailed,
    /// 光线起点已在视界内。
    OriginInsideHorizon,
    /// 无效质量等域错误。
    DomainError,
}

/// 一次完整追迹的诊断结果。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TraceDiagnostic {
    pub termination: RayTermination,
    pub steps: u32,
    /// 零测地线约束残差 |g_μν u^μ u^ν|（最后一次有效步）。
    pub null_residual: f64,
    pub disk_crossings: u32,
}

impl TraceDiagnostic {
    /// 终止分类伪彩色（诊断图）。
    pub fn termination_color(self) -> (u8, u8, u8) {
        match self.termination {
            RayTermination::Captured => (0, 0, 0),
            RayTermination::Escaped => (12, 24, 64),
            RayTermination::DiskCrossing => (220, 120, 40),
            RayTermination::StepBudgetExceeded => (200, 0, 200),
            RayTermination::NonFiniteState => (220, 0, 0),
            RayTermination::InitFailed => (220, 220, 0),
            RayTermination::OriginInsideHorizon => (32, 32, 32),
            RayTermination::DomainError => (128, 0, 128),
        }
    }

    /// 捕获轮廓：仅区分落入视界 / 逃逸 / 积分失败，不着色盘与星空。
    pub fn capture_silhouette_color(self) -> (u8, u8, u8) {
        match self.termination {
            RayTermination::Captured => (0, 0, 0),
            RayTermination::Escaped => (48, 52, 64),
            _ => self.termination_color(),
        }
    }

    /// 失败专用图：成功路径为中性灰，失败保留伪彩色。
    pub fn failure_only_color(self) -> (u8, u8, u8) {
        match self.termination {
            RayTermination::StepBudgetExceeded => (200, 0, 200),
            RayTermination::NonFiniteState => (220, 0, 0),
            RayTermination::InitFailed => (220, 220, 0),
            RayTermination::DomainError => (128, 0, 128),
            RayTermination::OriginInsideHorizon => (64, 64, 72),
            _ => (40, 42, 48),
        }
    }

    pub fn is_trace_failure(self) -> bool {
        matches!(
            self.termination,
            RayTermination::StepBudgetExceeded
                | RayTermination::NonFiniteState
                | RayTermination::InitFailed
                | RayTermination::DomainError
        )
    }

    pub fn to_rgb8(self, mode: DiagnosticColorMode) -> (u8, u8, u8) {
        match mode {
            DiagnosticColorMode::TerminationMap => self.termination_color(),
            DiagnosticColorMode::CaptureSilhouette => self.capture_silhouette_color(),
            DiagnosticColorMode::FailureOnly => self.failure_only_color(),
        }
    }
}

/// 诊断着色模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticColorMode {
    TerminationMap,
    CaptureSilhouette,
    /// 仅高亮积分失败像素，成功路径统一中性底。
    FailureOnly,
}

/// 整帧终止原因计数（诊断验收用）。
#[derive(Debug, Clone, Default)]
pub struct TerminationHistogram {
    counts: HashMap<RayTermination, u32>,
    total: u32,
}

impl TerminationHistogram {
    /// 记录一条光线的终止原因。
    pub fn record(&mut self, termination: RayTermination) {
        *self.counts.entry(termination).or_insert(0) += 1;
        self.total += 1;
    }

    /// 像素总数。
    pub fn total(&self) -> u32 {
        self.total
    }

    /// 某类终止的像素数。
    pub fn count(&self, termination: RayTermination) -> u32 {
        self.counts.get(&termination).copied().unwrap_or(0)
    }

    /// 某类终止占比（0–100）。
    pub fn percent(&self, termination: RayTermination) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.count(termination) as f64 / self.total as f64 * 100.0
    }

    /// 积分失败像素占比（预算耗尽 / 非有限 / 初始化失败 / 域错误）。
    pub fn trace_failure_percent(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let failed = self.count(RayTermination::StepBudgetExceeded)
            + self.count(RayTermination::NonFiniteState)
            + self.count(RayTermination::InitFailed)
            + self.count(RayTermination::DomainError);
        failed as f64 / self.total as f64 * 100.0
    }

    /// 按固定顺序输出人类可读摘要（stderr 用）。
    pub fn write_summary(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const ORDER: [RayTermination; 8] = [
            RayTermination::Captured,
            RayTermination::Escaped,
            RayTermination::DiskCrossing,
            RayTermination::StepBudgetExceeded,
            RayTermination::NonFiniteState,
            RayTermination::InitFailed,
            RayTermination::OriginInsideHorizon,
            RayTermination::DomainError,
        ];
        writeln!(f, "termination histogram ({} px):", self.total)?;
        for term in ORDER {
            let n = self.count(term);
            if n > 0 {
                writeln!(
                    f,
                    "  {:24} {:6} ({:5.1}%)",
                    termination_label(term),
                    n,
                    self.percent(term)
                )?;
            }
        }
        Ok(())
    }
}

impl fmt::Display for TerminationHistogram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.write_summary(f)
    }
}

fn termination_label(t: RayTermination) -> &'static str {
    match t {
        RayTermination::Captured => "Captured",
        RayTermination::Escaped => "Escaped",
        RayTermination::DiskCrossing => "DiskCrossing",
        RayTermination::StepBudgetExceeded => "StepBudgetExceeded",
        RayTermination::NonFiniteState => "NonFiniteState",
        RayTermination::InitFailed => "InitFailed",
        RayTermination::OriginInsideHorizon => "OriginInsideHorizon",
        RayTermination::DomainError => "DomainError",
    }
}
