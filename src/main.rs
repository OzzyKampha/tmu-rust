/// Print the crate version and a list of available example commands.
fn main() {
    println!("tmu-rs {} — weighted multiclass Tsetlin Machine", env!("CARGO_PKG_VERSION"));
    println!("Examples:");
    println!("  cargo run --release --example xor");
    println!("  cargo run --release --example noisy_xor");
    println!("  cargo run --release --example interpretability");
    println!("  cargo run --release --example ndr_flows");
}
