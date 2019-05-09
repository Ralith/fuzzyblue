//! Real-time physically based atmospheric scattering rendering
//!
//! Based on [E. Bruneton and F. Neyret's "Precomputed atmospheric
//! scattering"](https://ebruneton.github.io/precomputed_atmospheric_scattering/)

#![allow(clippy::missing_safety_doc)]

mod precompute;
pub use precompute::{Atmosphere, Builder, Parameters, PendingAtmosphere};

mod render;
pub use render::{DrawParameters, Renderer};
