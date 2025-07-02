from autosetup_ml.utils import *
# Mock tooth id lists (replace with your real values if needed)
# up_teeth_nums14 = [17, 16, 15, 14, 13, 12, 11, 21, 22,
#                     23, 24, 25, 26, 27]  
# dw_teeth_nums14 = [37, 36, 35, 34, 33, 32, 31, 41, 42,
#                     43, 44, 45, 46, 47] 

vals = dw_teeth_nums14 + up_teeth_nums14


def transforms_to_sorted(teeth_transforms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Return a list of transforms sorted by tooth ID (ascending).
    The output is the same structure (list of dicts) as the input.
    """
    # Собираем пары (tooth_id, transform)
    tooth_ids = dw_teeth_nums14 + up_teeth_nums14
    pairs = list(zip(tooth_ids, teeth_transforms))
    # Сортируем по tooth_id (на всякий случай, если порядок нарушен)
    pairs_sorted = sorted(pairs, key=lambda x: x[0])
    # Возвращаем только список transform в отсортированном порядке
    return [transform for _, transform in pairs_sorted]

def test_transforms_to_sorted():
    # Example tooth IDs (should match your dw_teeth_nums14 + up_teeth_nums14)
    tooth_ids = dw_teeth_nums14 + up_teeth_nums14
    # Create transforms with a marker for each tooth_id
    transforms = [{"tooth_id": tid, "val": f"val_{tid}"} for tid in tooth_ids]
    # Shuffle the transforms to simulate unordered input
    import random
    random.shuffle(transforms)
    # Run the sorting function
    sorted_transforms = transforms_to_sorted(transforms)
    # Check that the order matches the original tooth_ids
    for idx, t in enumerate(sorted_transforms):
        assert t["tooth_id"] == tooth_ids[idx], f"Mismatch at idx {idx}: got {t['tooth_id']}, expected {tooth_ids[idx]}"
    print("test_transforms_to_sorted passed.")

if __name__ == "__main__":
    test_transforms_to_sorted()