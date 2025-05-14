use pyo3::prelude::*;
#[pyfunction]
fn python_test_function(){
    println!("Hello from rust!");
}

#[pymodule]
fn trees(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(python_test_function, m)?)?;
    Ok(())
}

