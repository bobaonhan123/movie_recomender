import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.app.app import vectorize_user_profile

# Test vector hóa dữ liệu người dùng
def test_vectorization():
    # Test case 1: Nam, Hải Châu, student, education, 19-25
    vector1 = vectorize_user_profile("Nam", "Hải Châu", "student", "education", "19-25")
    print(f"Test 1 - Vector length: {len(vector1)}")
    print(f"Vector sum: {sum(vector1)} (should be 4 for the selected categories)")
    
    # Test case 2: Nữ, Cẩm Lệ, white collar, computer, 26-35
    vector2 = vectorize_user_profile("Nữ", "Cẩm Lệ", "white collar", "computer", "26-35")
    print(f"Test 2 - Vector length: {len(vector2)}")
    print(f"Vector sum: {sum(vector2)} (should be 3 for the selected categories, gender_Nam = 0)")
    
    # Print column names and their values for test 1
    x_knn_column_names = [
        'slot_A', 'slot_B', 'slot_C', 'slot_D', 'slot_E', 'slot_F', 'slot_G', 'slot_H', 'slot_I', 'slot_J', 'slot_K',
        'slot type_ĐÔI', 'popcorn_0', 'gender_Nam',
        'address_Cẩm Lệ', 'address_Huyện Hòa Vang', 'address_Hải Châu', 'address_III thành phố Đà Nẵng cũ',
        'address_Liên Chiểu', 'address_Ngũ Hành Sơn', 'address_Thanh Khê', 'address_công', 'address_Điểm cuối',
        'address_ủy quận Nhì Đà Nẵng', 'address_nan',
        'job_blue collar', 'job_specialist', 'job_student', 'job_white collar',
        'industry_computer', 'industry_construction', 'industry_economics', 'industry_education',
        'industry_engineering', 'industry_finance', 'industry_government agent', 'industry_health service',
        'industry_social service',
        'age_group_0-18', 'age_group_19-25', 'age_group_26-35', 'age_group_36-50', 'age_group_51-65'
    ]
    
    print(f"\nTest 1 - Active features:")
    for i, (col, val) in enumerate(zip(x_knn_column_names, vector1)):
        if val == 1:
            print(f"  {col}: {val}")
    
    print(f"\nTest 2 - Active features:")
    for i, (col, val) in enumerate(zip(x_knn_column_names, vector2)):
        if val == 1:
            print(f"  {col}: {val}")

if __name__ == "__main__":
    test_vectorization()
