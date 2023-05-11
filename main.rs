use loss_functions::Container;

mod loss_functions;

fn main() {
    println!("Hello, world!");
    let c = Container::new(loss_functions::MyType {});
    c.run();
    
}
